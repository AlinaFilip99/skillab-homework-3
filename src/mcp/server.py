

from __future__ import annotations

import asyncio
import sys

import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents import Orchestrator, AnalystAgent
from src.guardrails import PromptValidator


# ═════════════════════════════════════════════════════════════════════════
#  PARTEA DE SERVER  —  structura din Slide 10
#  (Server → handlers → stdio run loop)
# ═════════════════════════════════════════════════════════════════════════

# 1. SERVER — numele apare în serverInfo la `initialize` (Slide 5 & 6)
server = Server("document-analyst")


# 2. HANDLERS — declară ce expune serverul
@server.list_tools()                                  # răspunde la tools/list (Slide 7)
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_documents",
            description="Caută documente după un query text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Textul căutat"},
                },
                "required": ["query"],
            },
        ), 
        types.Tool(
            name="analyse_data",
            description="Analizeaza datele existente si returneaza informatii relevante.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Informatiile cerute"},
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()                                   # răspunde la tools/call (Slide 8)
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    # Logging pe stderr, NICIODATĂ pe stdout — stdout e doar JSON-RPC (Slide 3).
    print(f"[server] tools/call → {name} {arguments}", file=sys.stderr)

    if name == "search_documents":
        query = arguments["query"]
        # validate query
        v = PromptValidator(use_llm=False)
        r = v.validate(query)

        if not r.passed:
            return ValueError(f"Tool necunoscut: {name}")

        orch = Orchestrator()
        result = orch.run(query)
        return result

    elif name == "analyse_data":
        query = arguments["query"]
        # validate query
        v = PromptValidator(use_llm=False)
        r = v.validate(query)

        if not r.passed:
            return ValueError(f"Tool necunoscut: {name}")
        
        agent = AnalystAgent(tables_config={
            "achizitii_directe": {"schema_path": "data/schemas/achizitii_directe.json"},
            "anunturi_initiere": {"schema_path": "data/schemas/anunturi_initiere.json"},
        })
        result = agent.run(query)
        return result


    # Excepția devine automat un JSON-RPC error response (Slide 9).
    raise ValueError(f"Tool necunoscut: {name}")


# 3. RUN — loop-ul stdio (Slide 3 & 10)
async def run_server() -> None:
    async with stdio_server() as (read, write):
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name="document-analyst",
                server_version="1.0.0",
                capabilities=server.get_capabilities(   # capabilities din Slide 6
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


# ═════════════════════════════════════════════════════════════════════════
#  PARTEA DE CLIENT  —  narează lifecycle-ul (Slide 5) pas cu pas
# ═════════════════════════════════════════════════════════════════════════

def banner(label: str) -> None:
    print(f"\n{'═' * 60}\n  {label}\n{'═' * 60}")


async def run_client() -> None:
    # Clientul lansează serverul ca subprocess (Slide 1 & 3: "spawn").
    # Aici re-lansăm acest fișier cu flag-ul --server.
    params = StdioServerParameters(command=sys.executable, args=[__file__, "--server"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:

            # Lifecycle pas 1-3: initialize + initialized (Slide 5)
            banner("1. initialize  (handshake)")
            init = await session.initialize()
            print(f"   serverInfo  : {init.serverInfo.name} v{init.serverInfo.version}")
            print(f"   capabilities: {init.capabilities.model_dump(exclude_none=True)}")

            # tools/list — round-trip 1 (Slide 7 & 8)
            banner("2. tools/list  (ce poate face serverul?)")
            tools = (await session.list_tools()).tools
            for t in tools:
                print(f"   • {t.name} — {t.description}")

            # tools/call — round-trip 2 (Slide 8)
            banner("3. tools/call  search_documents(query='2024')")
            result = await session.call_tool("search_documents", {"query": "2024"})
            print(f"   result: {result.content[0].text}")

            # Error de protocol (Slide 9) — MCP întoarce isError=True, nu aruncă
            banner("4. tools/call  pe un tool inexistent  →  JSON-RPC error")
            err = await session.call_tool("nu_exista", {})
            print(f"   isError: {err.isError}")
            print(f"   mesaj  : {err.content[0].text}")

    banner("shutdown  (conexiune închisă)")


# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # --server → joacă rolul de server în subprocess; altfel → rulează demo-ul
    if "--server" in sys.argv:
        asyncio.run(run_server())
    else:
        asyncio.run(run_client())
