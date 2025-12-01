"""
Test script to verify the Data Monkey pipeline works end-to-end
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from pipeline.pipeline_orchestrator import PipelineOrchestrator
import pandas as pd


def test_pipeline():
    """Test the complete pipeline with test data"""
    print("🐵 Testing Data Monkey Pipeline")
    print("=" * 50)

    # Use test data
    test_file = "test_data.csv"
    if not os.path.exists(test_file):
        print(f"❌ Error: {test_file} not found")
        return False

    print(f"✓ Found test data: {test_file}")

    # Load data to verify it's valid
    try:
        df = pd.read_csv(test_file)
        print(f"✓ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False

    # Create pipeline orchestrator
    job_id = "test_job_001"
    print(f"\n✓ Creating pipeline orchestrator (job_id: {job_id})")

    try:
        orchestrator = PipelineOrchestrator(job_id, test_file)
        print("✓ Orchestrator created successfully")
    except Exception as e:
        print(f"❌ Error creating orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Execute pipeline
    print("\n" + "=" * 50)
    print("Running Pipeline Stages...")
    print("=" * 50)

    try:
        # Stage 1: Data Understanding
        print("\n[1/4] 🔍 Data Understanding...")
        result1 = orchestrator.execute_data_understanding(
            user_prompt="Analyze this dataset and identify the best target variable"
        )
        print(f"✓ Completed: Found {len(result1.get('column_analysis', {}))} columns")
        print(f"  Suggested target: {result1.get('problem_type', {}).get('suggested_target_column', 'N/A')}")
        print(f"  Problem type: {result1.get('problem_type', {}).get('problem_type', 'N/A')}")

        # Stage 2: Preprocessing
        print("\n[2/4] 🧹 Preprocessing...")
        result2 = orchestrator.execute_preprocessing(
            target_column=orchestrator.state.get('target_column')
        )
        print(f"✓ Completed: {result2.get('original_shape')} → {result2.get('final_shape')}")
        print(f"  Steps performed: {len(result2.get('steps_performed', []))}")

        # Stage 3: Model Selection
        print("\n[3/4] 🤖 Model Selection...")
        problem_type = orchestrator.state.get('problem_type', 'classification')
        result3 = orchestrator.execute_model_selection(problem_type)
        print(f"✓ Completed: Trained {len(result3.get('models_trained', []))} models")
        if result3.get('best_model'):
            print(f"  Best model: {result3['best_model']['model_name']}")
            metrics = result3['best_model'].get('metrics', {})
            if problem_type == 'classification':
                print(f"  Test accuracy: {metrics.get('test_accuracy', 0):.4f}")
            else:
                print(f"  Test R²: {metrics.get('test_r2', 0):.4f}")

        # Stage 4: Hyperparameter Tuning
        print("\n[4/4] ⚙️  Hyperparameter Tuning...")
        result4 = orchestrator.execute_hyperparameter_tuning(problem_type)
        if result4.get('status') == 'success':
            print(f"✓ Completed: Best score: {result4.get('best_score', 0):.4f}")
            print(f"  Best params: {result4.get('best_params', {})}")
        else:
            print(f"⚠️  Tuning status: {result4.get('status')}")

        print("\n" + "=" * 50)
        print("✅ Pipeline Execution Complete!")
        print("=" * 50)

        # Print final state summary
        state = orchestrator.get_state()
        print(f"\nFinal State Summary:")
        print(f"  Job ID: {state.get('job_id')}")
        print(f"  Current Stage: {state.get('current_stage')}")
        print(f"  Target Column: {state.get('target_column')}")
        print(f"  Problem Type: {state.get('problem_type')}")
        print(f"  Best Model: {state.get('best_model')}")

        # Check stage statuses
        print(f"\nStage Statuses:")
        for stage in ['data_understanding', 'preprocessing', 'model_selection', 'hyperparameter_tuning']:
            status = state.get(stage, {}).get('status', 'unknown')
            print(f"  {stage}: {status}")

        print("\n✅ All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    success = test_pipeline()
    print("\n")

    if success:
        print("🎉 Data Monkey Pipeline Test: PASSED")
        sys.exit(0)
    else:
        print("❌ Data Monkey Pipeline Test: FAILED")
        sys.exit(1)
