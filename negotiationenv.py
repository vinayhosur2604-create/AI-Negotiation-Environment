import random

# Environment Class
class NegotiationEnv:

    def __init__(self):
        self.reset()

    def reset(self):
        self.buyer_budget = random.randint(50, 100)
        self.seller_min_price = random.randint(30, 80)
        self.current_offer = random.randint(30, 100)
        self.done = False
        return self.get_state()

    def get_state(self):
        return (self.buyer_budget, self.seller_min_price, self.current_offer)

    def step(self, action):
        reward = 0

        if action == "increase":
            self.current_offer += 5

        elif action == "decrease":
            self.current_offer -= 5

        elif action == "accept":
            if self.seller_min_price <= self.current_offer <= self.buyer_budget:
                reward = 10
            else:
                reward = -5
            self.done = True

        elif action == "reject":
            reward = -10
            self.done = True

        return self.get_state(), reward, self.done


# Agent Logic
actions = ["increase", "decrease", "accept", "reject"]

def simple_agent(state):
    buyer_budget, seller_price, offer = state

    # Smart logic
    if offer < seller_price:
        return "increase"
    elif offer > buyer_budget:
        return "decrease"
    elif seller_price <= offer <= buyer_budget:
        return "accept"
    else:
        return random.choice(actions)


# MAIN PROGRAM
env = NegotiationEnv()
episodes = 5

for ep in range(episodes):
    state = env.reset()
    print(f"\n===== Negotiation {ep+1} =====")
    print(f"Buyer Budget: {state[0]}, Seller Min Price: {state[1]}, Starting Offer: {state[2]}")

    step_count = 0  # Prevent infinite loop

    while True:
        action = simple_agent(state)
        next_state, reward, done = env.step(action)

        print(f"Action: {action} | New Offer: {next_state[2]} | Reward: {reward}")

        state = next_state
        step_count += 1

        # Stop conditions
        if done or step_count > 20:
            print("Negotiation Finished")
            break