import asyncio
from get_from_url import get_stuff

async def main():
    url = 'https://2ch.hk/b/res/292696528.html'
    bytes_staff = await get_stuff(url)
    print(bytes_staff)

if __name__ == '__main__':
        asyncio.run(main())