import subprocess
program_list=['gpt_ent_str_distill_ecbd.py','gpt_ent_str_distill_ecbd2.py','gpt_ent_str_distill_ecbd3.py','gpt_ent_str_distill_ecbd4.py','gpt_ent_str_distill_ecbd5.py','gpt_ent_str_distill_ecbd6.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)
