from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, GraphRelationship, NodeLabel, RelType
from axon.core.storage.base import SearchResult
from axon.mcp.resources import get_dead_code_list, get_overview, get_schema
from axon.mcp.tools import (
    MAX_CYPHER_LENGTH,
    MAX_DIFF_LENGTH,
    _confidence_tag,
    _format_query_results,
    _group_by_process,
    handle_call_path,
    handle_communities,
    handle_concurrent_with,
    handle_context,
    handle_coupling,
    handle_cycles,
    handle_cypher,
    handle_dead_code,
    handle_detect_changes,
    handle_explain,
    handle_file_context,
    handle_impact,
    handle_list_repos,
    handle_query,
    handle_review_risk,
    handle_test_impact,
)


@pytest.fixture
def mock_storage():
    """Create a mock storage backend with common default return values."""
    storage = MagicMock()
    storage.fts_search.return_value = [
        SearchResult(
            node_id="function:src/auth.py:validate",
            score=1.0,
            node_name="validate",
            file_path="src/auth.py",
            label="function",
            snippet="def validate(user): ...",
        ),
    ]
    storage.get_node.return_value = GraphNode(
        id="function:src/auth.py:validate",
        label=NodeLabel.FUNCTION,
        name="validate",
        file_path="src/auth.py",
        start_line=10,
        end_line=30,
    )
    storage.get_callers.return_value = []
    storage.get_callees.return_value = []
    storage.get_type_refs.return_value = []
    storage.vector_search.return_value = []
    storage.traverse.return_value = []
    storage.traverse_with_depth.return_value = []
    storage.get_callers_with_confidence.return_value = []
    storage.get_callees_with_confidence.return_value = []
    storage.get_process_memberships.return_value = {}
    storage.execute_raw.return_value = []
    storage.get_accessors.return_value = []
    return storage


@pytest.fixture
def mock_storage_with_relations(mock_storage):
    """Storage mock with callers, callees, and type refs populated."""
    _caller = GraphNode(
        id="function:src/routes/auth.py:login_handler",
        label=NodeLabel.FUNCTION,
        name="login_handler",
        file_path="src/routes/auth.py",
        start_line=12,
        end_line=40,
    )
    _callee = GraphNode(
        id="function:src/auth/crypto.py:hash_password",
        label=NodeLabel.FUNCTION,
        name="hash_password",
        file_path="src/auth/crypto.py",
        start_line=5,
        end_line=20,
    )
    mock_storage.get_callers.return_value = [_caller]
    mock_storage.get_callees.return_value = [_callee]
    mock_storage.get_callers_with_confidence.return_value = [(_caller, 1.0)]
    mock_storage.get_callees_with_confidence.return_value = [(_callee, 0.8)]
    mock_storage.get_callers_with_metadata.return_value = [(_caller, 1.0, {})]
    mock_storage.get_callees_with_metadata.return_value = [(_callee, 0.8, {})]
    mock_storage.get_type_refs.return_value = [
        GraphNode(
            id="class:src/models.py:User",
            label=NodeLabel.CLASS,
            name="User",
            file_path="src/models.py",
            start_line=1,
            end_line=50,
        ),
    ]
    return mock_storage


class TestHandleListRepos:
    def test_no_registry_dir(self, tmp_path):
        result = handle_list_repos(registry_dir=tmp_path / "nonexistent")
        assert "No indexed repositories found" in result

    def test_empty_registry_dir(self, tmp_path):
        registry = tmp_path / "repos"
        registry.mkdir()
        result = handle_list_repos(registry_dir=registry)
        assert "No indexed repositories found" in result

    def test_with_repos(self, tmp_path):
        registry = tmp_path / "repos"
        repo_dir = registry / "my-project"
        repo_dir.mkdir(parents=True)
        meta = {
            "name": "my-project",
            "path": "/home/user/my-project",
            "stats": {
                "files": 25,
                "symbols": 150,
                "relationships": 200,
            },
        }
        (repo_dir / "meta.json").write_text(json.dumps(meta))

        result = handle_list_repos(registry_dir=registry)
        assert "my-project" in result
        assert "150" in result
        assert "200" in result
        assert "Indexed repositories (1)" in result


class TestHandleQuery:
    def test_returns_results(self, mock_storage):
        result = handle_query(mock_storage, "validate")
        assert "validate" in result
        assert "Function" in result
        assert "src/auth.py" in result
        assert "Next:" in result

    def test_no_results(self, mock_storage):
        mock_storage.fts_search.return_value = []
        mock_storage.vector_search.return_value = []
        result = handle_query(mock_storage, "nonexistent")
        assert "No results found" in result

    def test_snippet_included(self, mock_storage):
        result = handle_query(mock_storage, "validate")
        assert "def validate" in result

    def test_custom_limit(self, mock_storage):
        handle_query(mock_storage, "validate", limit=5)
        # hybrid_search calls fts_search with candidate_limit = limit * 3
        mock_storage.fts_search.assert_called_once_with("validate", limit=15)


class TestHandleContext:
    def test_basic_context(self, mock_storage):
        result = handle_context(mock_storage, "validate")
        assert "Symbol: validate (Function)" in result
        assert "src/auth.py:10-30" in result
        assert "Next:" in result

    def test_not_found_fts_empty(self, mock_storage):
        mock_storage.exact_name_search.return_value = []
        mock_storage.fts_search.return_value = []
        result = handle_context(mock_storage, "nonexistent")
        assert "not found" in result.lower()

    def test_not_found_node_none(self, mock_storage):
        mock_storage.get_node.return_value = None
        result = handle_context(mock_storage, "validate")
        assert "not found" in result.lower()

    def test_with_callers_callees_type_refs(self, mock_storage_with_relations):
        result = handle_context(mock_storage_with_relations, "validate")
        assert "Callers (1):" in result
        assert "login_handler" in result
        assert "Callees (1):" in result
        assert "hash_password" in result
        assert "Type references (1):" in result
        assert "User" in result

    def test_dead_code_flag(self, mock_storage):
        mock_storage.get_node.return_value = GraphNode(
            id="function:src/old.py:deprecated",
            label=NodeLabel.FUNCTION,
            name="deprecated",
            file_path="src/old.py",
            start_line=1,
            end_line=5,
            is_dead=True,
        )
        result = handle_context(mock_storage, "deprecated")
        assert "DEAD CODE" in result

    def test_heritage_shown(self, mock_storage):
        mock_storage.get_node.return_value = GraphNode(
            id="class:src/models.py:Admin",
            label=NodeLabel.CLASS,
            name="Admin",
            file_path="src/models.py",
            start_line=50,
            end_line=80,
        )
        mock_storage.execute_raw.side_effect = [
            # Heritage query
            [["User", "src/models.py", "extends"]],
            # Imported-by query
            [],
        ]
        result = handle_context(mock_storage, "Admin")
        assert "Heritage" in result
        assert "extends" in result
        assert "User" in result

    def test_imported_by_shown(self, mock_storage):
        mock_storage.execute_raw.side_effect = [
            # Heritage query
            [],
            # Imported-by query
            [["src/mcp/server.py"], ["tests/test_auth.py"]],
        ]
        result = handle_context(mock_storage, "validate")
        assert "Imported by (2)" in result
        assert "src/mcp/server.py" in result
        assert "tests/test_auth.py" in result


class TestHandleImpact:
    def test_no_downstream(self, mock_storage):
        result = handle_impact(mock_storage, "validate")
        assert "No upstream callers found" in result or "No downstream dependencies" in result

    def test_with_affected_symbols(self, mock_storage):
        _login = GraphNode(
            id="function:src/api.py:login",
            label=NodeLabel.FUNCTION,
            name="login",
            file_path="src/api.py",
            start_line=5,
            end_line=20,
        )
        _register = GraphNode(
            id="function:src/api.py:register",
            label=NodeLabel.FUNCTION,
            name="register",
            file_path="src/api.py",
            start_line=25,
            end_line=50,
        )
        mock_storage.traverse.return_value = [_login, _register]
        mock_storage.traverse_with_depth.return_value = [(_login, 1), (_register, 2)]
        mock_storage.get_callers_with_confidence.return_value = [(_login, 1.0)]
        result = handle_impact(mock_storage, "validate", depth=2)
        assert "Impact analysis for: validate" in result
        assert "Total: 2 symbols" in result
        assert "login" in result
        assert "register" in result
        assert "Depth: 2" in result

    def test_symbol_not_found(self, mock_storage):
        mock_storage.exact_name_search.return_value = []
        mock_storage.fts_search.return_value = []
        result = handle_impact(mock_storage, "nonexistent")
        assert "not found" in result.lower()


class TestHandleDeadCode:
    def test_no_dead_code(self, mock_storage):
        result = handle_dead_code(mock_storage)
        assert "No dead code detected" in result

    def test_with_dead_code(self, mock_storage):
        mock_storage.execute_raw.return_value = [
            ["function:src/old.py:unused_func", "unused_func", "src/old.py", 10, "Function"],
            ["class:src/models.py:DeprecatedModel", "DeprecatedModel", "src/models.py", 5, "Class"],
        ]
        result = handle_dead_code(mock_storage)
        assert "Dead Code Report (2 symbols)" in result
        assert "unused_func" in result
        assert "DeprecatedModel" in result

    def test_execute_raw_exception(self, mock_storage):
        mock_storage.execute_raw.side_effect = RuntimeError("DB error")
        result = handle_dead_code(mock_storage)
        assert "Could not retrieve dead code list" in result


SAMPLE_DIFF = """\
diff --git a/src/auth.py b/src/auth.py
index abc1234..def5678 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,5 +10,7 @@ def validate(user):
     if not user:
         return False
+    # Added new validation
+    check_permissions(user)
     return True
"""


class TestHandleDetectChanges:
    def test_parses_diff(self, mock_storage):
        # handle_detect_changes now uses execute_raw() with a Cypher query
        # to find symbols in the changed file.
        mock_storage.execute_raw.return_value = [
            ["function:src/auth.py:validate", "validate", "src/auth.py", 10, 30],
        ]

        result = handle_detect_changes(mock_storage, SAMPLE_DIFF)
        assert "src/auth.py" in result
        assert "validate" in result
        assert "Total affected symbols:" in result

    def test_empty_diff(self, mock_storage):
        result = handle_detect_changes(mock_storage, "")
        assert "Empty diff provided" in result

    def test_unparseable_diff(self, mock_storage):
        result = handle_detect_changes(mock_storage, "just some random text")
        assert "Could not parse" in result

    def test_no_symbols_in_changed_lines(self, mock_storage):
        mock_storage.execute_raw.return_value = []
        result = handle_detect_changes(mock_storage, SAMPLE_DIFF)
        assert "src/auth.py" in result
        assert "no indexed symbols" in result


class TestHandleCypher:
    def test_returns_results(self, mock_storage):
        mock_storage.execute_raw.return_value = [
            ["validate", "src/auth.py", 10],
            ["login", "src/api.py", 5],
        ]
        result = handle_cypher(mock_storage, "MATCH (n) RETURN n.name, n.file_path, n.start_line")
        assert "Results (2 rows)" in result
        assert "validate" in result
        assert "src/api.py" in result

    def test_no_results(self, mock_storage):
        result = handle_cypher(mock_storage, 'MATCH (n:Nonexistent) RETURN n')
        assert 'no results' in result.lower()

    def test_query_error(self, mock_storage, caplog):
        """Exception detail is not exposed to the caller; ref id links log."""
        mock_storage.execute_raw.side_effect = RuntimeError('Syntax error')
        with caplog.at_level('ERROR'):
            result = handle_cypher(mock_storage, 'INVALID QUERY')
        assert 'Syntax error' not in result
        assert 'failed' in result.lower()
        assert 'ref ' in result
        ref = result.split('ref ')[1].split(')')[0]
        matching = [
            r
            for r in caplog.records
            if r.exc_info and 'Syntax error' in str(r.exc_info[1])
        ]
        assert matching, 'Exception text must appear in a log record'
        assert matching[0].ref == ref

    def test_handle_cypher_rejects_write(self, mock_storage):
        result = handle_cypher(mock_storage, "DELETE (n)")
        assert "not permitted" in result.lower() or "not allowed" in result.lower()
        mock_storage.execute_raw.assert_not_called()


class TestResources:
    def test_get_schema(self):
        result = get_schema()
        assert "Node Labels:" in result
        assert "Relationship Types:" in result
        assert "CALLS" in result
        assert "Function" in result

    def test_get_overview(self, mock_storage):
        mock_storage.execute_raw.return_value = [["Function", 42]]
        result = get_overview(mock_storage)
        assert "Axon Codebase Overview" in result

    def test_get_dead_code_list(self, mock_storage):
        mock_storage.execute_raw.return_value = [
            ["function:src/old.py:old_func", "old_func", "src/old.py", 10, "Function"],
        ]
        result = get_dead_code_list(mock_storage)
        assert "Dead Code Report" in result
        assert "old_func" in result

    def test_get_dead_code_list_empty(self, mock_storage):
        result = get_dead_code_list(mock_storage)
        assert "No dead code detected" in result


class TestConfidenceTag:
    def test_high_confidence(self):
        assert _confidence_tag(1.0) == ""
        assert _confidence_tag(0.95) == ""
        assert _confidence_tag(0.9) == ""

    def test_medium_confidence(self):
        assert _confidence_tag(0.89) == " (~)"
        assert _confidence_tag(0.5) == " (~)"
        assert _confidence_tag(0.7) == " (~)"

    def test_low_confidence(self):
        assert _confidence_tag(0.49) == " (?)"
        assert _confidence_tag(0.1) == " (?)"
        assert _confidence_tag(0.0) == " (?)"


class TestConfidenceInContext:
    def test_medium_confidence_tag_shown(self, mock_storage_with_relations):
        result = handle_context(mock_storage_with_relations, "validate")
        # _callee has confidence 0.8, which produces " (~)"
        assert "(~)" in result

    def test_high_confidence_no_tag(self, mock_storage_with_relations):
        result = handle_context(mock_storage_with_relations, "validate")
        # login_handler has confidence 1.0 — no tag after its line
        assert "login_handler" in result
        # There should be no "(?)" for the high-confidence caller
        lines = result.split("\n")
        caller_line = [line for line in lines if "login_handler" in line][0]
        assert "(?)" not in caller_line
        assert "(~)" not in caller_line


class TestGroupByProcess:
    def test_empty_results(self, mock_storage):
        groups = _group_by_process([], mock_storage)
        assert groups == {}

    def test_with_memberships(self, mock_storage):
        results = [
            SearchResult(node_id="func:a", score=1.0, node_name="a"),
            SearchResult(node_id="func:b", score=0.9, node_name="b"),
            SearchResult(node_id="func:c", score=0.8, node_name="c"),
        ]
        mock_storage.get_process_memberships.return_value = {
            "func:a": "Auth Flow",
            "func:c": "Auth Flow",
        }
        groups = _group_by_process(results, mock_storage)
        assert "Auth Flow" in groups
        assert len(groups["Auth Flow"]) == 2

    def test_backend_missing_method(self, mock_storage):
        mock_storage.get_process_memberships.side_effect = AttributeError
        results = [SearchResult(node_id="func:a", score=1.0)]
        groups = _group_by_process(results, mock_storage)
        assert groups == {}


class TestFormatQueryResults:
    def test_ungrouped_only(self):
        results = [
            SearchResult(
                node_id="func:a", score=1.0, node_name="foo",
                file_path="src/a.py", label="function",
            ),
        ]
        output = _format_query_results(results, {})
        assert "foo (Function)" in output
        assert "src/a.py" in output
        assert "Next:" in output

    def test_with_groups(self):
        r1 = SearchResult(
            node_id="func:a", score=1.0, node_name="login",
            file_path="src/auth.py", label="function",
        )
        r2 = SearchResult(
            node_id="func:b", score=0.9, node_name="helper",
            file_path="src/utils.py", label="function",
        )
        groups = {"Auth Flow": [r1]}
        output = _format_query_results([r1, r2], groups)
        assert "=== Auth Flow ===" in output
        assert "=== Other results ===" in output
        assert "login" in output
        assert "helper" in output

    def test_snippet_truncation(self):
        long_snippet = "x" * 300
        results = [
            SearchResult(
                node_id="func:a", score=1.0, node_name="foo",
                file_path="src/a.py", label="function", snippet=long_snippet,
            ),
        ]
        output = _format_query_results(results, {})
        # Snippet in output should be at most 200 chars
        lines = output.split("\n")
        snippet_lines = [line for line in lines if line.strip().startswith("xxx")]
        for line in snippet_lines:
            assert len(line.strip()) <= 200


class TestImpactDepthGrouping:
    def test_depth_section_headers(self, mock_storage):
        _login = GraphNode(
            id="function:src/api.py:login",
            label=NodeLabel.FUNCTION,
            name="login",
            file_path="src/api.py",
            start_line=5,
            end_line=20,
        )
        _register = GraphNode(
            id="function:src/api.py:register",
            label=NodeLabel.FUNCTION,
            name="register",
            file_path="src/api.py",
            start_line=25,
            end_line=50,
        )
        mock_storage.traverse_with_depth.return_value = [
            (_login, 1), (_register, 2),
        ]
        mock_storage.get_callers_with_confidence.return_value = [(_login, 0.8)]

        result = handle_impact(mock_storage, "validate", depth=2)
        assert "Depth 1" in result
        assert "Direct callers (will break)" in result
        assert "Depth 2" in result
        assert "Indirect (may break)" in result

    def test_depth_3_transitive_label(self, mock_storage):
        _node = GraphNode(
            id="function:src/far.py:distant",
            label=NodeLabel.FUNCTION,
            name="distant",
            file_path="src/far.py",
            start_line=1,
            end_line=10,
        )
        mock_storage.traverse_with_depth.return_value = [(_node, 3)]
        mock_storage.get_callers_with_confidence.return_value = []

        result = handle_impact(mock_storage, "validate", depth=3)
        assert "Transitive (review)" in result

    def test_confidence_shown_for_direct_callers(self, mock_storage):
        _login = GraphNode(
            id="function:src/api.py:login",
            label=NodeLabel.FUNCTION,
            name="login",
            file_path="src/api.py",
            start_line=5,
            end_line=20,
        )
        mock_storage.traverse_with_depth.return_value = [(_login, 1)]
        mock_storage.get_callers_with_confidence.return_value = [(_login, 0.75)]

        result = handle_impact(mock_storage, "validate", depth=1)
        assert "confidence: 0.75" in result

    def test_depth_clamped_to_max(self, mock_storage):
        mock_storage.traverse_with_depth.return_value = []
        result = handle_impact(mock_storage, "validate", depth=100)
        assert "No upstream callers found" in result


class TestHandleCoupling:
    def test_returns_coupled_files(self, mock_storage):
        mock_storage.execute_raw.side_effect = [
            [["src/auth/session.py", 0.85, 12], ["src/tests/test_login.py", 0.72, 9]],
            [["src/auth/session.py"]],
        ]
        result = handle_coupling(mock_storage, "src/auth/login.py")
        assert "src/auth/session.py" in result
        assert "0.85" in result
        assert "src/tests/test_login.py" in result

    def test_flags_hidden_dependencies(self, mock_storage):
        mock_storage.execute_raw.side_effect = [
            [["src/tests/test_login.py", 0.72, 9]],
            [],
        ]
        result = handle_coupling(mock_storage, "src/auth/login.py")
        assert "hidden" in result.lower()

    def test_no_coupling_found(self, mock_storage):
        mock_storage.execute_raw.side_effect = [[], []]
        result = handle_coupling(mock_storage, "src/isolated.py")
        assert "No temporal coupling" in result

    def test_min_strength_filter(self, mock_storage):
        mock_storage.execute_raw.side_effect = [
            [["src/weak.py", 0.2, 2]],
            [],
        ]
        result = handle_coupling(mock_storage, "src/a.py", min_strength=0.3)
        assert "No temporal coupling" in result

    def test_empty_file_path(self, mock_storage):
        result = handle_coupling(mock_storage, '')
        assert 'required' in result.lower()

    def test_queries_use_code_relation(self, mock_storage):
        """Coupling queries use CodeRelation with rel_type, not bare table names."""
        mock_storage.execute_raw.side_effect = [
            [['src/auth/session.py', 0.85, 12]],
            [],
        ]
        handle_coupling(mock_storage, 'src/auth/login.py')

        all_queries = [
            str(call.args[0])
            for call in mock_storage.execute_raw.call_args_list
        ]
        coupling_queries = [q for q in all_queries if 'coupled_with' in q]
        assert coupling_queries, (
            'Expected at least one query referencing coupled_with'
        )
        for q in coupling_queries:
            assert 'CodeRelation' in q
            assert 'rel_type' in q
        assert not any('[:COUPLED_WITH]' in q for q in all_queries)


class TestHandleCommunities:
    def test_list_all_communities(self, mock_storage):
        mock_storage.execute_raw.side_effect = [
            [
                ["ingestion+storage", 0.72, '{"symbol_count": 23}'],
                ["mcp+server", 0.65, '{"symbol_count": 15}'],
            ],
            [],  # Cross-community processes
        ]
        result = handle_communities(mock_storage)
        assert "ingestion+storage" in result
        assert "0.72" in result
        assert "23" in result
        assert "mcp+server" in result

    def test_drill_into_community(self, mock_storage):
        mock_storage.execute_raw.return_value = [
            ["run_pipeline", "Function", "src/pipeline.py", 45, True, False],
            ["KuzuBackend", "Class", "src/storage.py", 10, False, True],
        ]
        result = handle_communities(mock_storage, community="ingestion+storage")
        assert "run_pipeline" in result
        assert "entry point" in result.lower()
        assert "KuzuBackend" in result

    def test_no_communities(self, mock_storage):
        mock_storage.execute_raw.side_effect = [[], []]
        result = handle_communities(mock_storage)
        assert "No communities" in result

    def test_community_not_found(self, mock_storage):
        mock_storage.execute_raw.return_value = []
        result = handle_communities(mock_storage, community='nonexistent')
        assert 'not found' in result.lower()

    def test_queries_use_code_relation(self, mock_storage):
        """Community queries use CodeRelation with rel_type, not bare table names."""
        mock_storage.execute_raw.side_effect = [
            [['ingestion+storage', 0.72, '{"symbol_count": 23}']],
            [],
        ]
        handle_communities(mock_storage)

        all_queries = [
            str(call.args[0])
            for call in mock_storage.execute_raw.call_args_list
        ]
        rel_queries = [
            q
            for q in all_queries
            if 'member_of' in q or 'step_in_process' in q
        ]
        assert rel_queries, (
            'Expected queries referencing member_of or step_in_process'
        )
        for q in rel_queries:
            assert 'CodeRelation' in q
            assert 'rel_type' in q
        assert not any('[:MEMBER_OF]' in q for q in all_queries)
        assert not any('[:STEP_IN_PROCESS]' in q for q in all_queries)


class TestHandleExplain:
    def test_basic_explanation(self, mock_storage):
        mock_storage.get_node.return_value = GraphNode(
            id="function:src/pipeline.py:run_pipeline",
            label=NodeLabel.FUNCTION,
            name="run_pipeline",
            file_path="src/pipeline.py",
            start_line=45,
            end_line=120,
            is_entry_point=True,
            is_exported=True,
        )
        mock_storage.get_callers_with_confidence.return_value = [
            (GraphNode(id="f:cli.py:main", label=NodeLabel.FUNCTION, name="main",
                       file_path="src/cli.py", start_line=1, end_line=10), 1.0),
        ]
        mock_storage.get_callees_with_confidence.return_value = [
            (GraphNode(id="f:walk.py:walk", label=NodeLabel.FUNCTION, name="walk",
                       file_path="src/walk.py", start_line=1, end_line=10), 0.9),
            (GraphNode(id="f:parse.py:parse", label=NodeLabel.FUNCTION, name="parse",
                       file_path="src/parse.py", start_line=1, end_line=10), 0.8),
        ]
        mock_storage.execute_raw.side_effect = [
            [["ingestion+storage"]],  # Community membership
            [["run → walk → parse", 1]],  # Process flows
        ]

        result = handle_explain(mock_storage, "run_pipeline")
        assert "run_pipeline" in result
        assert "Entry point" in result
        assert "Exported" in result
        assert "ingestion+storage" in result
        assert "Called by 1" in result
        assert "main" in result
        assert "Calls 2" in result

    def test_symbol_not_found(self, mock_storage):
        mock_storage.exact_name_search.return_value = []
        mock_storage.fts_search.return_value = []
        result = handle_explain(mock_storage, "nonexistent")
        assert "not found" in result.lower()

    def test_empty_symbol(self, mock_storage):
        result = handle_explain(mock_storage, "")
        assert "required" in result.lower()

    def test_dead_code_symbol(self, mock_storage):
        mock_storage.get_node.return_value = GraphNode(
            id="function:src/old.py:old_func",
            label=NodeLabel.FUNCTION,
            name="old_func",
            file_path="src/old.py",
            start_line=1,
            end_line=10,
            is_dead=True,
        )
        mock_storage.get_callers_with_confidence.return_value = []
        mock_storage.get_callees_with_confidence.return_value = []
        mock_storage.execute_raw.side_effect = [[], []]

        result = handle_explain(mock_storage, 'old_func')
        assert 'dead code' in result.lower() or 'Dead code' in result

    def test_queries_use_code_relation(self, mock_storage):
        """Explain queries use CodeRelation with rel_type, not bare table names."""
        mock_storage.get_node.return_value = GraphNode(
            id='function:src/pipeline.py:run_pipeline',
            label=NodeLabel.FUNCTION,
            name='run_pipeline',
            file_path='src/pipeline.py',
            start_line=45,
            end_line=120,
        )
        mock_storage.get_callers_with_confidence.return_value = []
        mock_storage.get_callees_with_confidence.return_value = []
        mock_storage.execute_raw.side_effect = [
            [['ingestion+storage']],
            [['run -> walk', 1]],
        ]

        handle_explain(mock_storage, 'run_pipeline')

        all_queries = [
            str(call.args[0])
            for call in mock_storage.execute_raw.call_args_list
        ]
        rel_queries = [
            q
            for q in all_queries
            if 'member_of' in q or 'step_in_process' in q
        ]
        assert rel_queries, (
            'Expected queries referencing member_of or step_in_process'
        )
        for q in rel_queries:
            assert 'CodeRelation' in q
            assert 'rel_type' in q
        assert not any('[:MEMBER_OF]' in q for q in all_queries)
        assert not any('[:STEP_IN_PROCESS]' in q for q in all_queries)


class TestHandleReviewRisk:
    def test_basic_risk_assessment(self, mock_storage):
        mock_storage.execute_raw.side_effect = [
            # Symbols in changed file
            [["function:src/auth.py:validate", "validate", "src/auth.py", 10, 30]],
            # Coupling for src/auth.py
            [["src/tests/test_auth.py", 0.82]],
            # Community for validate
            [["auth+security"]],
        ]
        mock_storage.get_node.return_value = GraphNode(
            id="function:src/auth.py:validate",
            label=NodeLabel.FUNCTION,
            name="validate",
            file_path="src/auth.py",
            start_line=10,
            end_line=30,
            is_entry_point=False,
        )
        mock_storage.traverse_with_depth.return_value = [
            (GraphNode(id="f:api.py:login", label=NodeLabel.FUNCTION, name="login",
                       file_path="src/api.py", start_line=5, end_line=20), 1),
        ]

        result = handle_review_risk(mock_storage, SAMPLE_DIFF)
        assert "Risk" in result
        assert "validate" in result

    def test_flags_missing_cochange_files(self, mock_storage):
        mock_storage.execute_raw.side_effect = [
            [["function:src/auth.py:validate", "validate", "src/auth.py", 10, 30]],
            [["src/tests/test_auth.py", 0.82]],
            [["auth"]],
        ]
        mock_storage.get_node.return_value = GraphNode(
            id="function:src/auth.py:validate",
            label=NodeLabel.FUNCTION,
            name="validate",
            file_path="src/auth.py",
            start_line=10,
            end_line=30,
        )
        mock_storage.traverse_with_depth.return_value = []
        result = handle_review_risk(mock_storage, SAMPLE_DIFF)
        assert "test_auth.py" in result
        assert "missing" in result.lower() or "usually change" in result.lower()

    def test_empty_diff(self, mock_storage):
        result = handle_review_risk(mock_storage, "")
        assert "Empty diff" in result

    def test_no_affected_symbols(self, mock_storage):
        mock_storage.execute_raw.side_effect = [
            [],  # No symbols in changed file
            [],  # No coupling
        ]
        result = handle_review_risk(mock_storage, SAMPLE_DIFF)
        assert 'No indexed symbols' in result or 'LOW' in result

    def test_queries_use_code_relation(self, mock_storage):
        """Review risk queries use CodeRelation with rel_type, not bare table names."""
        mock_storage.execute_raw.side_effect = [
            [
                [
                    'function:src/auth.py:validate',
                    'validate',
                    'src/auth.py',
                    10,
                    30,
                ]
            ],
            [['src/tests/test_auth.py', 0.82]],
            [['auth+security']],
        ]
        mock_storage.get_node.return_value = GraphNode(
            id='function:src/auth.py:validate',
            label=NodeLabel.FUNCTION,
            name='validate',
            file_path='src/auth.py',
            start_line=10,
            end_line=30,
        )
        mock_storage.traverse_with_depth.return_value = []

        handle_review_risk(mock_storage, SAMPLE_DIFF)

        all_queries = [
            str(call.args[0])
            for call in mock_storage.execute_raw.call_args_list
        ]
        coupling_queries = [q for q in all_queries if 'coupled_with' in q]
        assert coupling_queries, (
            'Expected at least one query referencing coupled_with'
        )
        for q in coupling_queries:
            assert 'CodeRelation' in q
            assert 'rel_type' in q

        member_of_queries = [q for q in all_queries if 'member_of' in q]
        assert member_of_queries, (
            'Expected at least one query referencing member_of'
        )
        for q in member_of_queries:
            assert 'CodeRelation' in q
            assert 'rel_type' in q

        assert not any('[:COUPLED_WITH]' in q for q in all_queries)
        assert not any('[:MEMBER_OF]' in q for q in all_queries)


class TestHandleCallPath:
    def test_direct_path(self, mock_storage):
        """Two symbols where A calls B directly."""
        _callee = GraphNode(
            id="function:src/perms.py:check_perms",
            label=NodeLabel.FUNCTION,
            name="check_perms",
            file_path="src/perms.py",
            start_line=25,
            end_line=40,
        )
        mock_storage.get_callees.return_value = [_callee]
        # Need separate fts_search results for from and to symbols
        mock_storage.fts_search.side_effect = [
            [
                SearchResult(
                    node_id='function:src/auth.py:validate',
                    score=1.0,
                    node_name='validate',
                )
            ],
            [
                SearchResult(
                    node_id='function:src/perms.py:check_perms',
                    score=1.0,
                    node_name='check_perms',
                )
            ],
        ]
        mock_storage.get_node.side_effect = [
            GraphNode(
                id='function:src/auth.py:validate',
                label=NodeLabel.FUNCTION,
                name='validate',
                file_path='src/auth.py',
                start_line=10,
                end_line=30,
            ),
            GraphNode(
                id='function:src/perms.py:check_perms',
                label=NodeLabel.FUNCTION,
                name='check_perms',
                file_path='src/perms.py',
                start_line=25,
                end_line=40,
            ),
            # get_node calls during path reconstruction
            GraphNode(
                id='function:src/auth.py:validate',
                label=NodeLabel.FUNCTION,
                name='validate',
                file_path='src/auth.py',
                start_line=10,
                end_line=30,
            ),
            GraphNode(
                id='function:src/perms.py:check_perms',
                label=NodeLabel.FUNCTION,
                name='check_perms',
                file_path='src/perms.py',
                start_line=25,
                end_line=40,
            ),
        ]
        result = handle_call_path(mock_storage, 'validate', 'check_perms')
        assert 'validate' in result
        assert 'check_perms' in result
        assert '1 hop' in result
        assert '->' in result

    def test_no_path_found(self, mock_storage):
        mock_storage.get_callees.return_value = []
        mock_storage.fts_search.side_effect = [
            [SearchResult(node_id="function:src/a.py:foo", score=1.0, node_name="foo")],
            [SearchResult(node_id="function:src/b.py:bar", score=1.0, node_name="bar")],
        ]
        mock_storage.get_node.side_effect = [
            GraphNode(id="function:src/a.py:foo", label=NodeLabel.FUNCTION,
                      name="foo", file_path="src/a.py", start_line=1, end_line=10),
            GraphNode(id="function:src/b.py:bar", label=NodeLabel.FUNCTION,
                      name="bar", file_path="src/b.py", start_line=1, end_line=10),
        ]
        result = handle_call_path(mock_storage, "foo", "bar")
        assert "No call path found" in result

    def test_same_symbol(self, mock_storage):
        mock_storage.fts_search.return_value = [
            SearchResult(node_id="function:src/a.py:foo", score=1.0, node_name="foo"),
        ]
        mock_storage.get_node.return_value = GraphNode(
            id="function:src/a.py:foo", label=NodeLabel.FUNCTION,
            name="foo", file_path="src/a.py", start_line=1, end_line=10,
        )
        result = handle_call_path(mock_storage, "foo", "foo")
        assert "same symbol" in result.lower()

    def test_empty_from_symbol(self, mock_storage):
        result = handle_call_path(mock_storage, "", "bar")
        assert "required" in result.lower()

    def test_empty_to_symbol(self, mock_storage):
        result = handle_call_path(mock_storage, "foo", "")
        assert "required" in result.lower()

    def test_source_not_found(self, mock_storage):
        mock_storage.exact_name_search.return_value = []
        mock_storage.fts_search.side_effect = [
            [],  # from_symbol not found
        ]
        result = handle_call_path(mock_storage, "nonexistent", "bar")
        assert "not found" in result.lower()


class TestHandleFileContext:
    def test_full_file_context(self, mock_storage):
        mock_storage.execute_raw.side_effect = [
            # Symbols
            [
                ["handle_query", "Function", 170, False, True, False],
                ["handle_context", "Function", 197, False, False, False],
            ],
            # Imports out
            [["src/storage/base.py"], ["src/search/hybrid.py"]],
            # Imported by
            [["src/mcp/server.py"]],
            # Coupling
            [["tests/mcp/test_tools.py", 0.85, 12]],
            # Dead code
            [],
            # Communities
            [['mcp+server', 2]],
            # Enums
            [],
            # Class attributes
            [],
            # Module constants
            [],
        ]
        result = handle_file_context(mock_storage, 'src/mcp/tools.py')
        assert 'src/mcp/tools.py' in result
        assert 'handle_query' in result
        assert 'entry point' in result.lower()
        assert 'Imports (2)' in result
        assert 'Imported by (1)' in result
        assert 'test_tools.py' in result
        assert '0.85' in result
        assert 'mcp+server' in result

    def test_empty_file(self, mock_storage):
        mock_storage.execute_raw.side_effect = [
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        ]
        result = handle_file_context(mock_storage, 'src/empty.py')
        assert 'No data found' in result

    def test_file_with_dead_code(self, mock_storage):
        mock_storage.execute_raw.side_effect = [
            [['old_func', 'Function', 45, True, False, False]],
            [],
            [],
            [],
            [['old_func', 45, 'Function']],
            [],
            [],
            [],
            [],
        ]
        result = handle_file_context(mock_storage, 'src/old.py')
        assert 'Dead code' in result
        assert 'old_func' in result

    def test_empty_file_path(self, mock_storage):
        result = handle_file_context(mock_storage, '')
        assert 'required' in result.lower()

    def test_queries_use_code_relation(self, mock_storage):
        """File context queries use CodeRelation with rel_type, not bare table names."""
        mock_storage.execute_raw.side_effect = [
            [['handle_query', 'Function', 170, False, True, False]],
            [['src/storage/base.py']],
            [['src/mcp/server.py']],
            [['tests/mcp/test_tools.py', 0.85, 12]],
            [],
            [['mcp+server', 2]],
            [],
            [],
            [],
        ]

        handle_file_context(mock_storage, 'src/mcp/tools.py')

        all_queries = [
            str(call.args[0])
            for call in mock_storage.execute_raw.call_args_list
        ]
        coupling_queries = [q for q in all_queries if 'coupled_with' in q]
        assert coupling_queries, (
            'Expected at least one query referencing coupled_with'
        )
        for q in coupling_queries:
            assert 'CodeRelation' in q
            assert 'rel_type' in q
        assert not any('[:COUPLED_WITH]' in q for q in all_queries)


class TestHandleTestImpact:
    def test_finds_test_callers_via_diff(self, mock_storage):
        # Changed symbol in diff
        mock_storage.execute_raw.return_value = [
            ["function:src/auth.py:validate", "validate", 10, 30],
        ]
        # Test function that calls validate
        _test_caller = GraphNode(
            id="function:tests/test_auth.py:test_validate",
            label=NodeLabel.FUNCTION,
            name="test_validate",
            file_path="tests/test_auth.py",
            start_line=5,
            end_line=15,
        )
        mock_storage.traverse_with_depth.return_value = [(_test_caller, 1)]

        result = handle_test_impact(mock_storage, diff=SAMPLE_DIFF)
        assert "test_validate" in result
        assert "tests/test_auth.py" in result
        assert "validate" in result

    def test_finds_test_callers_via_symbols(self, mock_storage):
        _test_caller = GraphNode(
            id="function:tests/test_auth.py:test_validate",
            label=NodeLabel.FUNCTION,
            name="test_validate",
            file_path="tests/test_auth.py",
            start_line=5,
            end_line=15,
        )
        mock_storage.traverse_with_depth.return_value = [(_test_caller, 1)]

        result = handle_test_impact(mock_storage, symbols=["validate"])
        assert "test_validate" in result
        assert "tests/test_auth.py" in result

    def test_no_tests_found(self, mock_storage):
        mock_storage.execute_raw.return_value = [
            ["function:src/auth.py:validate", "validate", 10, 30],
        ]
        # Non-test caller
        _caller = GraphNode(
            id="function:src/api.py:login",
            label=NodeLabel.FUNCTION,
            name="login",
            file_path="src/api.py",
            start_line=5,
            end_line=20,
        )
        mock_storage.traverse_with_depth.return_value = [(_caller, 1)]

        result = handle_test_impact(mock_storage, diff=SAMPLE_DIFF)
        assert "No test files found" in result

    def test_no_params(self, mock_storage):
        result = handle_test_impact(mock_storage)
        assert "provide either" in result.lower()

    def test_transitive_test(self, mock_storage):
        mock_storage.execute_raw.return_value = [
            ["function:src/auth.py:validate", "validate", 10, 30],
        ]
        _test_caller = GraphNode(
            id="function:tests/e2e/test_full.py:test_e2e",
            label=NodeLabel.FUNCTION,
            name="test_e2e",
            file_path="tests/e2e/test_full.py",
            start_line=5,
            end_line=15,
        )
        mock_storage.traverse_with_depth.return_value = [(_test_caller, 3)]

        result = handle_test_impact(mock_storage, diff=SAMPLE_DIFF)
        assert "indirect" in result.lower() or "transitive" in result.lower()
        assert "test_e2e" in result


class TestHandleCycles:
    def test_no_cycles(self, mock_storage):
        """Graph with no cycles returns clean message."""
        kg = KnowledgeGraph()
        # Add 3 nodes with no cycles: A -> B -> C
        a = GraphNode(id="function:a.py:a", label=NodeLabel.FUNCTION, name="a",
                      file_path="a.py", start_line=1, end_line=5)
        b = GraphNode(id="function:b.py:b", label=NodeLabel.FUNCTION, name="b",
                      file_path="b.py", start_line=1, end_line=5)
        c = GraphNode(id="function:c.py:c", label=NodeLabel.FUNCTION, name="c",
                      file_path="c.py", start_line=1, end_line=5)
        kg.add_node(a)
        kg.add_node(b)
        kg.add_node(c)
        kg.add_relationship(GraphRelationship(
            id="r1", type=RelType.CALLS, source=a.id, target=b.id))
        kg.add_relationship(GraphRelationship(
            id="r2", type=RelType.CALLS, source=b.id, target=c.id))

        mock_storage.load_graph.return_value = kg

        result = handle_cycles(mock_storage)
        assert "No circular dependencies" in result

    def test_detects_cycle(self, mock_storage):
        """Graph with A -> B -> A cycle is detected."""
        kg = KnowledgeGraph()
        a = GraphNode(id="function:a.py:a", label=NodeLabel.FUNCTION, name="a",
                      file_path="a.py", start_line=1, end_line=5)
        b = GraphNode(id="function:b.py:b", label=NodeLabel.FUNCTION, name="b",
                      file_path="b.py", start_line=1, end_line=5)
        kg.add_node(a)
        kg.add_node(b)
        kg.add_relationship(GraphRelationship(
            id="r1", type=RelType.CALLS, source=a.id, target=b.id))
        kg.add_relationship(GraphRelationship(
            id="r2", type=RelType.CALLS, source=b.id, target=a.id))

        mock_storage.load_graph.return_value = kg

        result = handle_cycles(mock_storage)
        assert "Circular Dependencies" in result
        assert "1 groups" in result or "Cycle 1" in result
        assert "a" in result
        assert "b" in result

    def test_critical_large_cycle(self, mock_storage):
        """Cycles with 5+ symbols are marked CRITICAL."""
        kg = KnowledgeGraph()
        nodes = []
        for i in range(5):
            n = GraphNode(id=f"function:{i}.py:f{i}", label=NodeLabel.FUNCTION,
                          name=f"f{i}", file_path=f"{i}.py", start_line=1, end_line=5)
            kg.add_node(n)
            nodes.append(n)
        # Create cycle: f0 -> f1 -> f2 -> f3 -> f4 -> f0
        for i in range(5):
            kg.add_relationship(GraphRelationship(
                id=f"r{i}", type=RelType.CALLS,
                source=nodes[i].id, target=nodes[(i + 1) % 5].id))

        mock_storage.load_graph.return_value = kg

        result = handle_cycles(mock_storage)
        assert 'CRITICAL' in result

    def test_load_graph_error(self, mock_storage, caplog):
        """Exception detail is not exposed to the caller; ref id links log."""
        mock_storage.load_graph.side_effect = RuntimeError('DB error')
        with caplog.at_level('ERROR'):
            result = handle_cycles(mock_storage)
        assert 'DB error' not in result
        assert 'ref ' in result
        ref = result.split('ref ')[1].split(')')[0]
        matching = [
            r
            for r in caplog.records
            if r.exc_info and 'DB error' in str(r.exc_info[1])
        ]
        assert matching, 'Exception text must appear in a log record'
        assert matching[0].ref == ref

    def test_empty_graph(self, mock_storage):
        kg = KnowledgeGraph()
        mock_storage.load_graph.return_value = kg
        result = handle_cycles(mock_storage)
        assert 'No symbols' in result


class TestInputCaps:
    """Input length cap enforcement across tool handlers."""

    def test_cypher_over_max_length_rejected(self, mock_storage):
        """Query exceeding MAX_CYPHER_LENGTH is rejected without executing."""
        query = 'MATCH ' + 'x' * (MAX_CYPHER_LENGTH + 1)
        result = handle_cypher(mock_storage, query)
        assert '100,000' in result
        mock_storage.execute_raw.assert_not_called()

    def test_cypher_at_max_length_allowed(self, mock_storage):
        """Query of exactly MAX_CYPHER_LENGTH characters proceeds to execute."""
        base = 'MATCH (n) RETURN n'
        query = base + 'x' * (MAX_CYPHER_LENGTH - len(base))
        assert len(query) == MAX_CYPHER_LENGTH
        mock_storage.execute_raw.return_value = []
        handle_cypher(mock_storage, query)
        mock_storage.execute_raw.assert_called_once()

    def test_diff_over_max_length_rejected_detect_changes(self, mock_storage):
        """Diff exceeding MAX_DIFF_LENGTH is rejected in handle_detect_changes."""
        diff = 'a' * (MAX_DIFF_LENGTH + 1)
        result = handle_detect_changes(mock_storage, diff)
        assert '100,000' in result
        mock_storage.execute_raw.assert_not_called()

    def test_diff_over_max_length_rejected_review_risk(self, mock_storage):
        """Diff exceeding MAX_DIFF_LENGTH is rejected in handle_review_risk."""
        diff = 'a' * (MAX_DIFF_LENGTH + 1)
        result = handle_review_risk(mock_storage, diff)
        assert '100,000' in result
        mock_storage.execute_raw.assert_not_called()

    def test_diff_over_max_length_rejected_test_impact(self, mock_storage):
        """Diff exceeding MAX_DIFF_LENGTH is rejected in handle_test_impact."""
        diff = 'a' * (MAX_DIFF_LENGTH + 1)
        result = handle_test_impact(mock_storage, diff=diff, symbols=None)
        assert '100,000' in result
        mock_storage.execute_raw.assert_not_called()


# ---------------------------------------------------------------------------
# Diff constant reused across repo_path integration tests.
# ---------------------------------------------------------------------------
_PY_DIFF = """\
diff --git a/src/foo.py b/src/foo.py
index 0000000..1111111 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -2,1 +2,1 @@ def bar():
-    \"\"\"Old docstring.\"\"\"
+    \"\"\"New docstring.\"\"\"
"""


class TestHandleTestImpactRepoPath:
    """Integration tests: handle_test_impact with repo_path wiring."""

    def test_no_repo_path_no_exception_no_warnings(self, mock_storage) -> None:
        """Existing callers without repo_path still work; no Warnings section."""
        mock_storage.execute_raw.return_value = [
            ['function:src/auth.py:validate', 'validate', 10, 30]
        ]
        mock_storage.traverse_with_depth.return_value = []
        result = handle_test_impact(mock_storage, diff=SAMPLE_DIFF)
        assert 'Warnings:' not in result
        assert 'Error' not in result

    def test_docstring_only_hunk_is_filtered(
        self, mock_storage, tmp_path: Path
    ) -> None:
        """Hunk covering only a Python docstring is listed under warnings.

        A diff that touches only a docstring line should cause the file to
        appear under 'Warnings: Ignored (docstring/comment-only changes):'.
        """
        src = tmp_path / 'src'
        src.mkdir()
        (src / 'foo.py').write_text(
            'def bar():\n    """Old docstring."""\n    pass\n'
        )
        mock_storage.execute_raw.return_value = []
        mock_storage.traverse_with_depth.return_value = []
        # Diff changes line 2 (the docstring) of src/foo.py.
        result = handle_test_impact(
            mock_storage, diff=_PY_DIFF, repo_path=tmp_path
        )
        assert 'Ignored (docstring/comment-only changes):' in result
        assert 'src/foo.py' in result

    def test_executable_hunk_not_in_warnings(
        self, mock_storage, tmp_path: Path
    ) -> None:
        """Hunk covering an executable line does not appear in the warnings."""
        src = tmp_path / 'src'
        src.mkdir()
        (src / 'foo.py').write_text('def bar():\n    x = 1\n    return x\n')
        executable_diff = """\
diff --git a/src/foo.py b/src/foo.py
index 0000000..1111111 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -2,1 +2,1 @@ def bar():
-    x = 1
+    x = 2
"""
        mock_storage.execute_raw.return_value = [
            ['function:src/foo.py:bar', 'bar', 1, 3]
        ]
        mock_storage.traverse_with_depth.return_value = []
        result = handle_test_impact(
            mock_storage, diff=executable_diff, repo_path=tmp_path
        )
        assert 'Ignored (docstring/comment-only changes):' not in result

    def test_testpaths_excludes_out_of_scope_test_file(
        self, mock_storage, tmp_path: Path
    ) -> None:
        """Test caller outside testpaths appears in config-excluded warnings."""
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.pytest.ini_options]\ntestpaths = ["tests/integration"]\n'
        )
        # Changed symbol in src.
        mock_storage.execute_raw.return_value = [
            ['function:src/auth.py:validate', 'validate', 10, 30]
        ]
        # Caller in tests/unit (outside testpaths).
        _unit_test = GraphNode(
            id='function:tests/unit/test_x.py:test_validate',
            label=NodeLabel.FUNCTION,
            name='test_validate',
            file_path='tests/unit/test_x.py',
            start_line=5,
            end_line=15,
        )
        mock_storage.traverse_with_depth.return_value = [(_unit_test, 1)]

        result = handle_test_impact(
            mock_storage, diff=SAMPLE_DIFF, repo_path=tmp_path
        )
        # Should NOT appear in affected tests.
        assert (
            'Affected tests' not in result
            or 'tests/unit/test_x.py' not in result
        )
        # Should appear in excluded-by-config warnings.
        assert 'Excluded by pytest config' in result
        assert 'tests/unit/test_x.py' in result

    def test_testpaths_keeps_in_scope_test_file(
        self, mock_storage, tmp_path: Path
    ) -> None:
        """Test caller inside testpaths appears in affected tests, not warnings."""
        (tmp_path / 'pyproject.toml').write_text(
            '[tool.pytest.ini_options]\ntestpaths = ["tests/integration"]\n'
        )
        mock_storage.execute_raw.return_value = [
            ['function:src/auth.py:validate', 'validate', 10, 30]
        ]
        _integ_test = GraphNode(
            id='function:tests/integration/test_y.py:test_validate',
            label=NodeLabel.FUNCTION,
            name='test_validate',
            file_path='tests/integration/test_y.py',
            start_line=5,
            end_line=15,
        )
        mock_storage.traverse_with_depth.return_value = [(_integ_test, 1)]

        result = handle_test_impact(
            mock_storage, diff=SAMPLE_DIFF, repo_path=tmp_path
        )
        assert 'tests/integration/test_y.py' in result
        assert 'Excluded by pytest config' not in result


class TestDottedPathIntegration:
    """Integration tests: dotted-path symbol names flow through tool handlers."""

    def test_handle_context_dotted_path(self, mock_storage) -> None:
        """handle_context resolves 'Foo.bar' via exact_name_search."""
        method_result = SearchResult(
            node_id='method:src/a.py:bar',
            score=2.0,
            node_name='bar',
            file_path='src/a.py',
            label='method',
            snippet='def bar(self): ...',
        )
        mock_storage.exact_name_search.return_value = [method_result]
        mock_storage.get_node.return_value = GraphNode(
            id='method:src/a.py:bar',
            label=NodeLabel.METHOD,
            name='bar',
            file_path='src/a.py',
            start_line=5,
            end_line=15,
        )

        result = handle_context(mock_storage, 'Foo.bar')

        mock_storage.exact_name_search.assert_called_once_with(
            'Foo.bar', limit=1
        )
        assert 'bar' in result
        assert 'src/a.py' in result

    def test_handle_call_path_dotted_from_symbol(self, mock_storage) -> None:
        """handle_call_path resolves 'Foo.bar' as the from-symbol via exact_name_search."""
        from_result = SearchResult(
            node_id='method:src/a.py:bar',
            score=2.0,
            node_name='bar',
            file_path='src/a.py',
            label='method',
        )
        to_result = SearchResult(
            node_id='function:src/b.py:baz',
            score=2.0,
            node_name='baz',
            file_path='src/b.py',
            label='function',
        )
        _from_node = GraphNode(
            id='method:src/a.py:bar',
            label=NodeLabel.METHOD,
            name='bar',
            file_path='src/a.py',
            start_line=5,
            end_line=15,
        )
        _to_node = GraphNode(
            id='function:src/b.py:baz',
            label=NodeLabel.FUNCTION,
            name='baz',
            file_path='src/b.py',
            start_line=1,
            end_line=10,
        )
        # exact_name_search resolves 'Foo.bar'; fts_search resolves 'baz'.
        mock_storage.exact_name_search.return_value = [from_result]
        mock_storage.fts_search.return_value = [to_result]
        mock_storage.get_node.side_effect = [_from_node, _to_node]
        mock_storage.get_callees.return_value = []

        result = handle_call_path(mock_storage, 'Foo.bar', 'baz')

        mock_storage.exact_name_search.assert_any_call('Foo.bar', limit=1)
        # Path not found (no call edge) but resolution succeeded.
        assert 'not found' in result.lower() or 'bar' in result


# ---------------------------------------------------------------------------
# Phase 4b tests
# ---------------------------------------------------------------------------


class TestHandleImpactPropagate:
    """handle_impact with propagate_through parameter."""

    def test_propagate_through_limits_traversal(self, mock_storage) -> None:
        """Only edges with matching dispatch_kind are traversed."""
        _direct = GraphNode(
            id='function:src/a.py:direct_caller',
            label=NodeLabel.FUNCTION,
            name='direct_caller',
            file_path='src/a.py',
            start_line=1,
            end_line=10,
        )
        _threaded = GraphNode(
            id='function:src/b.py:thread_caller',
            label=NodeLabel.FUNCTION,
            name='thread_caller',
            file_path='src/b.py',
            start_line=1,
            end_line=10,
        )
        # get_callers_with_metadata returns both callers; direct_caller has
        # dispatch_kind "direct" and thread_caller has "thread_executor".
        mock_storage.get_callers_with_metadata.return_value = [
            (_direct, 1.0, {'dispatch_kind': 'direct'}),
            (_threaded, 1.0, {'dispatch_kind': 'thread_executor'}),
        ]
        result = handle_impact(
            mock_storage, 'validate', propagate_through=['direct']
        )
        assert 'direct_caller' in result
        assert 'thread_caller' not in result

    def test_propagate_through_none_preserves_legacy_path(
        self, mock_storage
    ) -> None:
        """propagate_through=None uses traverse_with_depth (legacy path)."""
        _login = GraphNode(
            id='function:src/api.py:login',
            label=NodeLabel.FUNCTION,
            name='login',
            file_path='src/api.py',
            start_line=5,
            end_line=20,
        )
        mock_storage.traverse_with_depth.return_value = [(_login, 1)]
        mock_storage.get_callers_with_confidence.return_value = [(_login, 1.0)]

        result = handle_impact(
            mock_storage, 'validate', propagate_through=None
        )

        mock_storage.traverse_with_depth.assert_called_once()
        assert 'login' in result

    def test_propagate_through_empty_list_no_edges_followed(
        self, mock_storage
    ) -> None:
        """propagate_through=[] means nothing is followed."""
        mock_storage.get_callers_with_metadata.return_value = [
            (
                GraphNode(
                    id='function:src/a.py:some_caller',
                    label=NodeLabel.FUNCTION,
                    name='some_caller',
                    file_path='src/a.py',
                    start_line=1,
                    end_line=5,
                ),
                1.0,
                {'dispatch_kind': 'direct'},
            )
        ]
        result = handle_impact(mock_storage, 'validate', propagate_through=[])
        assert 'No upstream callers' in result or 'propagate_through' in result


class TestHandleConcurrentWith:
    """handle_concurrent_with tests."""

    def test_reaches_thread_executor_callback(self, mock_storage) -> None:
        """Thread-executor callees are reported under [thread_executor]."""
        _callback = GraphNode(
            id='function:src/tasks.py:callback',
            label=NodeLabel.FUNCTION,
            name='callback',
            file_path='src/tasks.py',
            start_line=10,
            end_line=20,
        )
        mock_storage.get_callees_with_metadata.return_value = [
            (_callback, 1.0, {'dispatch_kind': 'thread_executor'})
        ]
        result = handle_concurrent_with(mock_storage, 'validate')
        assert 'callback' in result
        assert 'thread_executor' in result

    def test_ignores_direct_edges(self, mock_storage) -> None:
        """Callees connected by direct edges are not reported."""
        _plain = GraphNode(
            id='function:src/utils.py:helper',
            label=NodeLabel.FUNCTION,
            name='helper',
            file_path='src/utils.py',
            start_line=1,
            end_line=5,
        )
        mock_storage.get_callees_with_metadata.return_value = [
            (_plain, 1.0, {})
        ]
        result = handle_concurrent_with(mock_storage, 'validate')
        assert 'No concurrently-dispatched callees found' in result

    def test_depth_bounded(self, mock_storage) -> None:
        """depth=1 limits traversal to direct neighbours only."""
        _callback = GraphNode(
            id='function:src/tasks.py:cb',
            label=NodeLabel.FUNCTION,
            name='cb',
            file_path='src/tasks.py',
            start_line=5,
            end_line=10,
        )
        # First call returns one async callee; subsequent calls return
        # nothing, ensuring we don't recurse.
        mock_storage.get_callees_with_metadata.side_effect = [
            [(_callback, 1.0, {'dispatch_kind': 'detached_task'})],
            [],
        ]
        result = handle_concurrent_with(mock_storage, 'validate', depth=1)
        assert 'cb' in result

    def test_symbol_not_found(self, mock_storage) -> None:
        """Returns appropriate message for unknown symbol."""
        mock_storage.exact_name_search.return_value = []
        mock_storage.fts_search.return_value = []
        result = handle_concurrent_with(mock_storage, 'nonexistent_xyz')
        assert 'not found' in result.lower()

    def test_mixed_dispatch_kinds_grouped(self, mock_storage) -> None:
        """Multiple dispatch_kinds produce separate grouped sections."""
        _cb1 = GraphNode(
            id='function:src/a.py:thread_cb',
            label=NodeLabel.FUNCTION,
            name='thread_cb',
            file_path='src/a.py',
            start_line=1,
            end_line=5,
        )
        _cb2 = GraphNode(
            id='function:src/b.py:detached_cb',
            label=NodeLabel.FUNCTION,
            name='detached_cb',
            file_path='src/b.py',
            start_line=1,
            end_line=5,
        )
        mock_storage.get_callees_with_metadata.return_value = [
            (_cb1, 1.0, {'dispatch_kind': 'thread_executor'}),
            (_cb2, 1.0, {'dispatch_kind': 'detached_task'}),
        ]
        result = handle_concurrent_with(mock_storage, 'validate')
        assert 'thread_executor' in result
        assert 'detached_task' in result
        assert 'thread_cb' in result
        assert 'detached_cb' in result


class TestHandleContextDispatchTags:
    """handle_context display enrichment with dispatch / metadata tags."""

    def test_displays_dispatch_kind_for_non_direct_callees(
        self, mock_storage
    ) -> None:
        """Callee with dispatch_kind='thread_executor' shows [thread_executor] tag."""
        _callee = GraphNode(
            id='function:src/tasks.py:worker',
            label=NodeLabel.FUNCTION,
            name='worker',
            file_path='src/tasks.py',
            start_line=5,
            end_line=15,
        )
        mock_storage.get_callers_with_metadata.return_value = []
        mock_storage.get_callees_with_metadata.return_value = [
            (_callee, 1.0, {'dispatch_kind': 'thread_executor'})
        ]
        result = handle_context(mock_storage, 'validate')
        assert '[thread_executor]' in result

    def test_omits_dispatch_kind_for_direct_edges(self, mock_storage) -> None:
        """Callees with default dispatch produce no extra dispatch tag."""
        _callee = GraphNode(
            id='function:src/utils.py:helper',
            label=NodeLabel.FUNCTION,
            name='helper',
            file_path='src/utils.py',
            start_line=1,
            end_line=5,
        )
        mock_storage.get_callers_with_metadata.return_value = []
        mock_storage.get_callees_with_metadata.return_value = [
            (_callee, 1.0, {})
        ]
        result = handle_context(mock_storage, 'validate')
        # No dispatch tag should appear for direct (default) edges.
        assert '[direct]' not in result
        assert '[thread_executor]' not in result

    def test_displays_awaited_and_in_try(self, mock_storage) -> None:
        """Callee with awaited=True and in_try=True shows both tags."""
        _callee = GraphNode(
            id='function:src/io.py:fetch',
            label=NodeLabel.FUNCTION,
            name='fetch',
            file_path='src/io.py',
            start_line=1,
            end_line=10,
        )
        mock_storage.get_callers_with_metadata.return_value = []
        mock_storage.get_callees_with_metadata.return_value = [
            (_callee, 1.0, {'awaited': True, 'in_try': True})
        ]
        result = handle_context(mock_storage, 'validate')
        assert '[awaited]' in result
        assert '[in_try]' in result

    def test_displays_return_consumption_for_callers(
        self, mock_storage
    ) -> None:
        """Caller with return_consumption='passed_through' shows tag."""
        _caller = GraphNode(
            id='function:src/pipeline.py:pipe',
            label=NodeLabel.FUNCTION,
            name='pipe',
            file_path='src/pipeline.py',
            start_line=1,
            end_line=10,
        )
        mock_storage.get_callers_with_metadata.return_value = [
            (_caller, 1.0, {'return_consumption': 'passed_through'})
        ]
        mock_storage.get_callees_with_metadata.return_value = []
        result = handle_context(mock_storage, 'validate')
        assert 'passed_through' in result

    def test_omits_return_consumption_for_stored(self, mock_storage) -> None:
        """Default return_consumption='stored' produces no tag."""
        _caller = GraphNode(
            id='function:src/a.py:caller',
            label=NodeLabel.FUNCTION,
            name='caller',
            file_path='src/a.py',
            start_line=1,
            end_line=10,
        )
        mock_storage.get_callers_with_metadata.return_value = [
            (_caller, 1.0, {'return_consumption': 'ignored'})
        ]
        mock_storage.get_callees_with_metadata.return_value = []
        result = handle_context(mock_storage, 'validate')
        assert '[return: ignored]' not in result


class TestHandleCallPathDispatchAnnotation:
    """handle_call_path hop dispatch annotation."""

    def test_hop_dispatch_annotation(self, mock_storage) -> None:
        """Hop with dispatch_kind='detached_task' shows annotation on the hop line."""
        _mid = GraphNode(
            id='function:src/dispatch.py:dispatch',
            label=NodeLabel.FUNCTION,
            name='dispatch',
            file_path='src/dispatch.py',
            start_line=10,
            end_line=20,
        )
        _target = GraphNode(
            id='function:src/worker.py:run',
            label=NodeLabel.FUNCTION,
            name='run',
            file_path='src/worker.py',
            start_line=5,
            end_line=15,
        )
        mock_storage.fts_search.side_effect = [
            [
                SearchResult(
                    node_id='function:src/auth.py:validate',
                    score=1.0,
                    node_name='validate',
                )
            ],
            [
                SearchResult(
                    node_id='function:src/worker.py:run',
                    score=1.0,
                    node_name='run',
                )
            ],
        ]
        mock_storage.get_node.side_effect = [
            GraphNode(
                id='function:src/auth.py:validate',
                label=NodeLabel.FUNCTION,
                name='validate',
                file_path='src/auth.py',
                start_line=10,
                end_line=30,
            ),
            GraphNode(
                id='function:src/worker.py:run',
                label=NodeLabel.FUNCTION,
                name='run',
                file_path='src/worker.py',
                start_line=5,
                end_line=15,
            ),
            # During path reconstruction
            GraphNode(
                id='function:src/auth.py:validate',
                label=NodeLabel.FUNCTION,
                name='validate',
                file_path='src/auth.py',
                start_line=10,
                end_line=30,
            ),
            _mid,
            _target,
        ]
        # BFS: validate -> dispatch -> run
        mock_storage.get_callees.side_effect = [
            [_mid],  # callees of validate
            [_target],  # callees of dispatch
            [],
        ]
        # Metadata: dispatch-to-run edge has detached_task
        mock_storage.get_callees_with_metadata.side_effect = [
            [],  # validate -> dispatch: no metadata
            [
                (_target, 1.0, {'dispatch_kind': 'detached_task'})
            ],  # dispatch -> run
        ]
        result = handle_call_path(mock_storage, 'validate', 'run')
        assert '[detached_task]' in result
