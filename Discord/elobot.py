import os

import discord
from dotenv import load_dotenv

import rank_functions

#get token from our .env
load_dotenv()
token = os.getenv('elo_bot_token')

#initialize the client
client = discord.Client()

#let us know when the client has initialized
@client.event
async def on_ready():
    print(f'{client.user} is online')

#On member join
@client.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(
        f'Hey {member.name}, this feature works'
    )

#On message
@client.event
async def on_message(message):
    #Ignore messages sent by the bot
    if message.author == client.user:
        return
    
    #If the message starts with the identifier !rankbot
    if message.content.startswith('!rankbot'):
        msg= f'Hey {message.author.mention}'
        channel = message.channel
        p = rank_functions.parse_command(message)
        print(p)
        await channel.send(msg)


client.run(token)