f = open('inference.py', 'r', encoding='utf-8')
c = f.read()
f.close()

c = c.replace(
    'rewards=0.00',
    'rewards=0.01'
)
c = c.replace(
    'reward=0.00 done=true error=Reset failed',
    'reward=0.01 done=true error=Reset failed'
)
c = c.replace(
    'reward=0.00 done=true error=Step failed',
    'reward=0.01 done=true error=Step failed'
)

open('inference.py', 'w', encoding='utf-8').write(c)
print('DONE')
