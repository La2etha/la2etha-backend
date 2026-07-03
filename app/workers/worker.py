"""RQ worker entrypoint. Run with:  python -m app.workers.worker

Uses SimpleWorker because Windows lacks os.fork(); on Linux the default forking
Worker also works. ponytail: SimpleWorker runs jobs in-process (no fork
isolation) — fine for a single-GPU box; swap to Worker on a Linux deployment.
"""

from rq import SimpleWorker

from app.workers import QUEUE_NAME, get_redis


def main() -> None:
    connection = get_redis()
    worker = SimpleWorker([QUEUE_NAME], connection=connection)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
