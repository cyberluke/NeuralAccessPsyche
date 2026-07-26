"""Unit tests for WorkflowStore SQLite persistence.

Tests workflow CRUD operations, event logging, and artifact storage.
"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from core.agentic.contracts import (
    InnovationWorkflowState,
    WorkflowStatus,
    WorkflowEvent,
    SelectedDirection,
    Roadmap,
)
from core.agentic.workflow_store import WorkflowStore


def _make_state(workflow_id="wf-test", phase="repository_ingestion", status=WorkflowStatus.RUNNING):
    """Helper to create a valid InnovationWorkflowState."""
    return InnovationWorkflowState(
        workflow_id=workflow_id,
        repository_path="/test/repo",
        repository_revision="abc123",
        user_goal="Test goal",
        status=status,
        phase=phase,
    )


class TestWorkflowStore:
    """Test workflow persistence and retrieval."""

    def setup_method(self):
        """Create a temporary database for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test_workflows.db")
        self.store = WorkflowStore(db_path=self.db_path)

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_workflow(self):
        """Test creating a new workflow."""
        state = _make_state("wf-test-1")
        workflow_id = self.store.create_workflow(state)
        assert workflow_id == "wf-test-1"

    def test_get_workflow(self):
        """Test retrieving a workflow by ID."""
        state = _make_state("wf-test-2", phase="archaeologist")
        self.store.create_workflow(state)
        
        retrieved = self.store.get_workflow("wf-test-2")
        assert retrieved is not None
        assert retrieved.workflow_id == "wf-test-2"
        assert retrieved.status == WorkflowStatus.RUNNING
        assert retrieved.phase == "archaeologist"

    def test_get_nonexistent_workflow(self):
        """Test retrieving a workflow that doesn't exist."""
        retrieved = self.store.get_workflow("wf-nonexistent")
        assert retrieved is None

    def test_update_workflow(self):
        """Test updating workflow state."""
        state = _make_state("wf-test-3")
        self.store.create_workflow(state)
        
        # Update the workflow
        state.phase = "heretic"
        state.status = WorkflowStatus.WAITING_FOR_APPROVAL
        self.store.update_workflow(state)
        
        # Retrieve and verify
        retrieved = self.store.get_workflow("wf-test-3")
        assert retrieved.phase == "heretic"
        assert retrieved.status == WorkflowStatus.WAITING_FOR_APPROVAL

    def test_list_workflows(self):
        """Test listing all workflows."""
        for i in range(3):
            state = _make_state(f"wf-test-{i}")
            self.store.create_workflow(state)
        
        workflows = self.store.list_workflows()
        assert len(workflows) == 3
        workflow_ids = {w.workflow_id for w in workflows}
        assert workflow_ids == {"wf-test-0", "wf-test-1", "wf-test-2"}

    def test_add_event(self):
        """Test adding a workflow event."""
        state = _make_state("wf-test-4")
        self.store.create_workflow(state)
        
        event = WorkflowEvent(
            workflow_id="wf-test-4",
            sequence=1,
            event_type="phase_started",
            phase="repository_ingestion",
            summary="Started repository ingestion",
            timestamp=datetime.now(timezone.utc),
        )
        self.store.add_event(event)
        
        # get_events uses sequence > after_sequence (default 0)
        events = self.store.get_events("wf-test-4")
        assert len(events) == 1
        assert events[0].event_type == "phase_started"
        assert events[0].phase == "repository_ingestion"

    def test_get_events_ordered_by_sequence(self):
        """Test that events are returned in sequence order."""
        state = _make_state("wf-test-5")
        self.store.create_workflow(state)
        
        # Add events with sequences 1, 2, 3 (sequence > 0 is the default filter)
        for seq in [3, 1, 2]:
            event = WorkflowEvent(
                workflow_id="wf-test-5",
                sequence=seq,
                event_type=f"event_{seq}",
                phase="test_phase",
                summary=f"Event {seq}",
                timestamp=datetime.now(timezone.utc),
            )
            self.store.add_event(event)
        
        events = self.store.get_events("wf-test-5")
        assert len(events) == 3
        assert [e.sequence for e in events] == [1, 2, 3]

    def test_store_and_get_artifact(self):
        """Test storing and retrieving a workflow artifact."""
        state = _make_state("wf-test-6")
        self.store.create_workflow(state)
        
        artifact_data = {
            "direction": "Test direction",
            "rationale": "Test rationale",
            "evidence_ids": ["ev-1", "ev-2"],
        }
        self.store.store_artifact("wf-test-6", "art-1", "selected_direction", artifact_data)
        
        retrieved = self.store.get_artifact("art-1")
        assert retrieved is not None
        assert retrieved["direction"] == "Test direction"

    def test_get_artifacts_by_type(self):
        """Test retrieving artifacts filtered by type."""
        state = _make_state("wf-test-7")
        self.store.create_workflow(state)
        
        direction_data = {
            "direction": "Test direction",
            "rationale": "Test rationale",
        }
        roadmap_data = {
            "phases": [{"phase": "Phase 1", "tasks": ["Task 1"]}],
        }
        
        self.store.store_artifact("wf-test-7", "art-dir-1", "selected_direction", direction_data)
        self.store.store_artifact("wf-test-7", "art-road-1", "roadmap", roadmap_data)
        
        # Filter by type
        directions = self.store.get_artifacts_by_type("wf-test-7", "selected_direction")
        assert len(directions) == 1
        assert directions[0]["direction"] == "Test direction"
        
        roadmaps = self.store.get_artifacts_by_type("wf-test-7", "roadmap")
        assert len(roadmaps) == 1
        assert roadmaps[0]["phases"][0]["phase"] == "Phase 1"

    def test_workflow_persistence_across_store_instances(self):
        """Test that workflows persist when creating a new store instance."""
        state = _make_state("wf-test-8")
        self.store.create_workflow(state)
        
        # Create a new store instance pointing to the same database
        new_store = WorkflowStore(db_path=self.db_path)
        
        # Should be able to retrieve the workflow
        retrieved = new_store.get_workflow("wf-test-8")
        assert retrieved is not None
        assert retrieved.workflow_id == "wf-test-8"
