f = open('inference.py', 'r', encoding='utf-8')
c = f.read()
f.close()

old = "    rewards_str = \",\".join([f\"{round(min(0.99, max(0.01, r)), 2):.2f}\" for r in step_rewards])\n    print(f\"[END] success={str(done).lower()} steps={step} rewards={rewards_str}\", flush=True)\n    final_reward = step_rewards[-1] if step_rewards else 0.0\n    return final_reward if done else total_reward / max(step, 1)\n    raw = final_reward if done else total_reward / max(step, 1)\n    return round(min(0.99, max(0.01, raw)), 3)"

print("SEARCHING...")
print(old in c)
idx = c.find("rewards_str")
print(repr(c[idx:idx+400]))
