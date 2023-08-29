import asyncio
import fer

lst_0 = [
    "test_mov/test_1.mkv",
    "test_mov/test_2.mkv",
    "test_mov/test_3.mkv",
]
lst_1 = [
    "test_mov_2/test_1.avi",
    "test_mov_2/test_2.avi",
    "test_mov_2/test_3.avi",
]

async def main(uf_lst, pf_lst):
    await fer.main(uf_lst, pf_lst)

asyncio.run(main(lst_0, lst_1))