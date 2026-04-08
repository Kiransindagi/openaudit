f = open('inference.py', 'r', encoding='utf-8')
c = f.read()
f.close()

c = c.replace(
    'return 0.0\n\n    observation',
    'return 0.01\n\n    observation'
)
c = c.replace(
    'return final_reward if done else total_reward / max(step, 1)',
    'raw = final_reward if done else total_reward / max(step, 1)\n    return round(min(0.99, max(0.01, raw)), 3)'
)
open('inference.py', 'w', encoding='utf-8').write(c)
print('DONE')
