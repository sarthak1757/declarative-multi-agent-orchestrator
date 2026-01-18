import asyncio
import logging

class Executor:
    def __init__(self, agents):
        self.agents = agents
        self.context = ""

    async def sequential(self, steps):
        logging.info("Starting sequential workflow")
        for step in steps:
            agent = self.agents[step["agent"]]
            logging.info(f"Executing step with agent: {step['agent']}")
            self.context = await agent.run(self.context)
        logging.info("Sequential workflow finished")
        return self.context

    async def parallel(self, branches, then):
        logging.info(f"Starting parallel workflow with branches: {branches}")
        tasks = [self.agents[a].run(self.context) for a in branches]
        results = await asyncio.gather(*tasks)
        self.context = "\n".join(results)
        logging.info(f"Parallel branches finished. Executing 'then' step with agent: {then['agent']}")
        return await self.agents[then["agent"]].run(self.context)

    async def supervisor(self, root, subs):
        logging.info(f"Starting supervisor workflow (Fan-Out/Fan-In). Root: {root}, Sub-agents: {subs}")
        
        # Fan-Out: Create tasks for all sub-agents
        tasks = []
        for sub in subs:
            logging.info(f"Supervisor delegating to sub-agent: {sub}")
            tasks.append(self.agents[sub].run(self.context))
            
        # Wait for all sub-agents to complete
        results = await asyncio.gather(*tasks)
        
        # Fan-In: Aggregate results
        self.context = "\n".join(results)
        logging.info(f"All sub-agents finished. Executing root agent: {root}")
        return await self.agents[root].run(self.context)