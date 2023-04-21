import subprocess
program_list=['gpt_ft_distill_ecbd.py','gpt_ft_distill_ecbd2.py','gpt_ft_distill_ecbd3.py','gpt_ft_distill_ecbd4.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)
