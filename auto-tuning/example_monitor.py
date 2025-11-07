"""
실시간 모니터링 사용 예제

UI에 표시할 수 있는 데이터를 JSON 형식으로 제공합니다.
"""

from realtime_monitor import RealtimeMonitor
import json
import time


def example_console_monitoring():
    """콘솔 출력 예제"""
    from realtime_monitor import monitor_tuning

    print("=== 콘솔 모니터링 예제 ===")
    print("실시간으로 튜닝 상태를 모니터링합니다.")
    print()

    # 2초마다 상태 출력
    monitor_tuning("tuning_database", interval=2.0)


def example_json_output():
    """JSON 출력 예제 (UI 연동용)"""
    monitor = RealtimeMonitor("tuning_database")

    print("=== JSON 출력 예제 (UI 연동용) ===")
    print("튜닝 상태를 JSON 형식으로 출력합니다.")
    print()

    # 한 번만 상태 가져오기
    status_dict = monitor.get_status_dict()

    # Pretty print JSON
    print(json.dumps(status_dict, indent=2, ensure_ascii=False))

    print("\n\n이 JSON 데이터를 UI에 표시하세요:")
    print("- total_trials: 전체 trial 수")
    print("- completed_trials: 완료된 trial 수")
    print("- current_task: 현재 처리 중인 task")
    print("- recent_trials: 최근 trial 결과")
    print("- cost_model_updates: Cost model 학습 진행 상황")


def example_continuous_json():
    """지속적인 JSON 업데이트 (웹 서버 등에서 사용)"""
    monitor = RealtimeMonitor("tuning_database")

    print("=== 지속적인 JSON 업데이트 예제 ===")
    print("2초마다 JSON 상태를 출력합니다.")
    print("Ctrl+C로 종료하세요.")
    print()

    try:
        while True:
            status_dict = monitor.get_status_dict()

            # 간단한 요약 출력
            print(f"\n[{status_dict['last_update']}]")
            print(f"Progress: {status_dict['completed_trials']}/{status_dict['total_trials']} trials")

            if status_dict['current_task']:
                task = status_dict['current_task']
                print(f"Current Task: {task['task_name']}")
                print(f"  Trials: {task['trials_completed']}")
                if task['best_latency_us']:
                    print(f"  Best Latency: {task['best_latency_us']:.2f} us")

            # 최근 cost model 업데이트
            if status_dict['cost_model_updates']:
                latest = status_dict['cost_model_updates'][-1]
                print(f"Latest Cost Model: iter {latest['iteration']}, RMSE: {latest['tr_rmse']:.6f}")

            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n\n모니터링 종료.")


def example_ui_data_format():
    """UI에 표시할 데이터 형식 설명"""
    print("=== UI에 표시할 데이터 ===\n")

    print("1. 전체 진행률 (Progress Bar)")
    print("   - total_trials: 전체 trial 수")
    print("   - completed_trials: 완료된 trial 수")
    print("   - Progress: (completed_trials / total_trials) * 100%")
    print()

    print("2. Task 진행 상황 (Table)")
    print("   - all_tasks: 모든 task 리스트")
    print("   - 각 task:")
    print("     - task_name: Task 이름")
    print("     - trials_completed: 완료된 trial 수")
    print("     - best_latency_us: 최적 latency (microseconds)")
    print("     - speed_gflops: 계산 속도 (GFLOPS)")
    print("     - is_done: 완료 여부")
    print()

    print("3. Cost Model 학습 그래프 (Line Chart)")
    print("   - cost_model_updates: Cost model 업데이트 리스트")
    print("   - 각 업데이트:")
    print("     - iteration: XGBoost iteration 번호")
    print("     - tr_rmse: Training RMSE (낮을수록 좋음)")
    print("     - tr_a_peak_32: Accuracy at peak@32 (높을수록 좋음)")
    print()

    print("4. 최근 Trial 결과 (Table)")
    print("   - recent_trials: 최근 trial 결과 리스트")
    print("   - 각 trial:")
    print("     - workload_id: Workload ID")
    print("     - runtime_ms: 실행 시간 (milliseconds)")
    print("     - trace_length: 스케쥴 변환 단계 수")
    print()

    print("5. 실시간 로그 (Log Viewer)")
    print("   - logs/task_scheduler.log 파일을 tail -f로 읽기")
    print("   - 또는 realtime_monitor의 상태를 주기적으로 폴링")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("사용법:")
        print("  python example_monitor.py console    # 콘솔 모니터링")
        print("  python example_monitor.py json       # JSON 한 번 출력")
        print("  python example_monitor.py continuous # 지속적인 JSON 출력")
        print("  python example_monitor.py ui         # UI 데이터 형식 설명")
        print()
        mode = "ui"

    if mode == "console":
        example_console_monitoring()
    elif mode == "json":
        example_json_output()
    elif mode == "continuous":
        example_continuous_json()
    elif mode == "ui":
        example_ui_data_format()
    else:
        print(f"알 수 없는 모드: {mode}")
        print("console, json, continuous, ui 중 하나를 선택하세요.")
