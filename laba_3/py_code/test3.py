import hashlib
from itertools import product
from multiprocessing import Pool, cpu_count

unhashed_phones_list = [89167621860,89716321126,89350099076,89115978074,89215561294]
original_phones_list = [89156651327,89868999648,89639261431,89837232568,89099367499]
original_hash_list = ['48847b4b82f40447582e170d7b3b8e0e','00f162fae89b11e04f94016fc752ccef','07c171e5aa3d2d8ac0b490a3d2d9dfb6','f775ae5a96c82579ec130c4e36deccc8','b48c6d40f4751318314467874406b6d7']
salts_list = [-10970533,152678522,289162355,721254494,-116193795]

def run_md5(string):
    h = hashlib.md5(string.encode()).hexdigest()
    return h

def check_hash(hash,original_hash_list):
    if(hash in original_hash_list):
        print(f'{hash} in list')
        return True

def calc_salt():
    for i in range(len(original_phones_list)):
        cur_salt = abs(original_phones_list[i] - unhashed_phones_list[i])
        
        
        new_phone1 = str(unhashed_phones_list[i] + cur_salt)
        new_phone2 = str(unhashed_phones_list[i] - cur_salt)
        new_phone3 = str(original_phones_list[i] + cur_salt)
        new_phone4 = str(original_phones_list[i] - cur_salt)
        
        hash1 = run_md5(new_phone1)
        hash2 = run_md5(new_phone2)
        hash3 = run_md5(new_phone3)
        hash4 = run_md5(new_phone4)
        

        print(hash1)
        print(hash2)
        print(hash3)
        print(hash4)
        
        if(hash1 in original_hash_list):
            print(f'hash1 in list, {hash1}')
        if(hash2 in original_hash_list):
            print(f'hash2 in list, {hash2}')
        if(hash3 in original_hash_list):
            print(f'hash3 in list, {hash3}')
        if(hash4 in original_hash_list):
            print(f'hash4 in list, {hash4}')
            
            
        print('\n\n')

def test_salts():
    for salt in salts_list:
        i = 0
        for i in range(len(unhashed_phones_list)):
            new_phone = str(original_phones_list[i] - salt)
            #new_phone2 = str(unhashed_phones_list[i] - salt)
            #new_phone3 = str(original_phones_list[i] + salt)
            #new_phone4 = str(original_phones_list[i] - salt)
            
            hash = run_md5(new_phone)
            # hash2 = run_md5(new_phone2)
            # hash3 = run_md5(new_phone3)
            # hash4 = run_md5(new_phone4)
            if(hash in original_hash_list):
                i += 1
        if(i==5):
            print(f'salt - {salt} is suitable')

test_salts()
# -116193795 -> abs(-116193795) + phone_number_unhashed -> hash -> original hash