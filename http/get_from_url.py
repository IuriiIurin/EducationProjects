import asyncio
from http.client import HTTPException
from httpx import AsyncClient, TimeoutException, ProtocolError

async def get_raw_http_result(url: str, http_client: AsyncClient):
    try:
        responce = await http_client.get(url)
    except TimeoutException:
        raise HTTPException(f'Fail to ger answer from "{url}"')
    except ProtocolError:
        raise HTTPException(f'Wrong url "{url}"')

    if responce.is_error:
        raise HTTPException(f'Server returned Err. Url: {url} Err: {responce.content}.')

    return responce.content

async def make_something(bytes_stuff):
    return bytes_stuff.decode('utf-8')

async def get_stuff(url: str):
    bytes_stuff = await get_raw_http_result(url, AsyncClient())
    stuff = await make_something(bytes_stuff)

    return stuff


