import subprocess
program_list=['gpt_ft_ecbd_1ep.py', 'gpt_ft_ecbd_2ep.py', 'gpt_ft_ecbd_3ep.py', 'gpt_ft_ecbd_4ep.py', 'gpt_ft_ecbd_5ep.py', 'gpt_ft_ecbd_6ep.py', 'gpt_ft_ecbd_7ep.py', 'gpt_ft_ecbd_8ep.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)
