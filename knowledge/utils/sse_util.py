import asyncio
import json
import logging
import queue
import threading
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:
    PROGRESS = "progress"  # 任务节点进度
    DELTA = "delta"  # LLM 流式输出增量
    FINAL = "final"  # 最终完整答案


# 全局 SSE 任务队列存储
# Key: task_id, Value: queue.Queue
_task_stream: Dict[str, queue.Queue] = {}
_stream_lock = threading.Lock()


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    """获取指定任务的队列（线程安全）"""
    with _stream_lock:
        return _task_stream.get(task_id)


def create_sse_queue(task_id: str) -> queue.Queue:
    """创建并注册一个新的 SSE 队列（线程安全）"""
    q = queue.Queue()
    with _stream_lock:
        _task_stream[task_id] = q
    return q


def remove_sse_queue(task_id: str):
    """移除指定任务的队列（线程安全）
    不存在 key 默认返回 None
    """
    with _stream_lock:
        _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_sse_event(task_id: str, event: str, data: Dict[str, Any]):
    """
    通过 task_id 推送事件到 SSE 队列
    """
    # 1. 获取 SSE 队列
    stream_queue = get_sse_queue(task_id)

    # 2. 队列存在则推送，否则记录警告日志
    if stream_queue:
        stream_queue.put({"event": event, "data": data})
        logging.debug(f"SSE push success: task={task_id[-8:]}, event={event}")
    else:
        logging.warning(f"SSE queue not found for task_id={task_id}, event={event} dropped")


async def sse_generator(task_id: str, request: Request) -> AsyncGenerator:
    """
    流式输出结果的消费者
    1. 从sse队列中获取结果
    2. 封装队列中的数据以及事件类型为sse协议的数据包格式
    3. 将封装好的数据包yield出去
    4. 收到 FINAL 事件后主动结束，不再空转等待
    Args:
        task_id: 任务id

    Returns:
        AsyncGenerator:异步生成器对象

    """

    # 1. 根据任务id 获取任务队列对象（线程安全）
    sse_queue = get_sse_queue(task_id)

    # 2. 校验
    if sse_queue is None:
        logging.warning(f"sse_generator: queue not found for task_id={task_id}")
        return

    loop = asyncio.get_event_loop()

    # 3. 让当前线程一直从队列中获取数据【如果队列一旦有数据，就直接获取，如果队列没有数据，等一会，在问一下】
    try:
        while True:

            # 3.1 判断前端sse连接是否关闭（主动探测）--->FastApI:可以感知到：request
            if await request.is_disconnected():
                return
            try:
                # 3.2 从队列中获取(阻塞队列---)为了让事件循环不阻塞，
                msg = await loop.run_in_executor(None, sse_queue.get, True, 1)
                # 3.3 获取事件类型
                event_type = msg.get('event')
                # 3.4 获取事件数据
                event_data = msg.get('data')
                # 3.5 打包返回
                yield _sse_pack(event_type, event_data)  # 打包并且通过yield返回

                # 3.6 收到 FINAL 事件后主动结束，不再空转
                if event_type == SSEEvent.FINAL:
                    return
            except queue.Empty:
                continue
    except  (ConnectionResetError, BrokenPipeError) as e:
        # 客户端中断 关闭了窗口或者浏览器
        return

    except asyncio.CancelledError:
        # 服务端中断 协程被取消 重新抛出，让外层知道它被成功取消()
        raise
    finally:
        remove_sse_queue(task_id)
