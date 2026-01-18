import logging

class Agent:
    def __init__(self, id, role, goal, llm, tools):
        self.id = id
        self.role = role
        self.goal = goal
        self.llm = llm
        self.tools = tools

    async def run(self, context: str):
        logging.info(f"Agent {self.id} starting. Role: {self.role}")
        prompt = f"Role: {self.role}\nGoal: {self.goal}\nContext:\n{context}"
        response = await self.llm.generate(prompt, self.tools)
        logging.info(f"Agent {self.id} finished.")
        return response