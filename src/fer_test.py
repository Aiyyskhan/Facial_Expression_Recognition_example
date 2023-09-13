from datetime import datetime
import asyncio

# import fer
import fer_2

start_time = datetime.now()

lst_0 = [
    # "test_mov/test_1.mkv",
    # "test_mov/test_2.mkv",
    # "test_mov/test_3.mkv",
    0
]
lst_1 = [
    "test_mov_2/test_1.mp4",
    # "test_mov_2/test_1.avi",
    # "test_mov_2/test_2.avi",
    # "test_mov_2/test_3.avi",
]

async def main(uf_lst, pf_lst):
    # await fer.main(uf_lst, pf_lst)
    await fer_2.main(uf_lst, pf_lst)

asyncio.run(main(lst_0, lst_1))

print(f"Time: {datetime.now() - start_time}")