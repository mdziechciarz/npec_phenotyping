from celery import Celery
from database import get_db, Task
from datetime import datetime, timezone
from fastapi import Depends
from sqlalchemy.orm import Session

app = Celery('tasks', broker='redis://redis:6379/0')

@app.task
def process_batch_task(task_id: int):
    db = next(get_db())
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        return
    
    now_utc = datetime.now(timezone.utc)
    delay = (task.start_time - now_utc).total_seconds()
    if delay > 0:
        time.sleep(delay)

    # Processing logic here
    # e.g., processing files associated with the task
    task.status = "completed"
    db.commit()
    db.close()
