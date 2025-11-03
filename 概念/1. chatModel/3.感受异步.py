import asyncio
import  time
def call_model():
    print("开始调用模型")
    time.sleep(5)
    print("调用模型结束")
def work_other_work():
    for i in range(5):
        print(f"正在开始其他任务{i}")
        time.sleep(1)
def main():
    start_time = time.time()
    call_model()
    work_other_work()
    end_time = time.time()
    print(f"总花费{end_time - start_time}")

async  def async_call_model():
    await asyncio.sleep(5)
    print("模型调用完成")
async def async_work_other_work():
    await asyncio.sleep(5)
    print("其他任务完成")

async  def async_main():
    start_time = time.time()
    await  asyncio.gather(async_call_model(), async_work_other_work())
    end_time = time.time()
    print(f"总花费{end_time - start_time}")


main()
asyncio.run(async_main())