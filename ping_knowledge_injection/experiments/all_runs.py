import subprocess
program_list=['t5_ping_entity_inferences.py','t5_ping_entity_inferences_2.py','t5_ping_entity_inferences_3.py','t5_ping_entity_inferences_4.py','t5_ping_entity_inferences_5.py']

for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)
