# coding: utf-8

import re
import requests
import subprocess

# OJ
class OnlineJudge(object):
    def __init__(self, language="python") -> None:
        self.language = language
        self.solution_dir = f"../LeetCode-Solutions/{self.language}"
        self.leetcode_code_dir = "/home/nus_ids_user3/.leetcode/code"
        self.leetcode_map = self.get_leetcode_map()
    
    @staticmethod
    def get_leetcode_map():
        url = "https://leetcode.com/api/problems/all/"
        payload = {}
        headers = {
            'authority': 'leetcode.cn',
            'accept': '*/*',
            'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh-HK;q=0.6,zh-TW;q=0.5,zh;q=0.4',
            'content-type': 'application/json',
            'cookie': '_gid=GA1.2.1098388302.1701430581; gr_user_id=7410146c-e5bc-4739-a7c8-40cb0e580219; a2873925c34ecbd2_gr_session_id=5c028fe5-e819-4177-bc61-685d662a8ea3; a2873925c34ecbd2_gr_session_id_sent_vst=5c028fe5-e819-4177-bc61-685d662a8ea3; Hm_lvt_f0faad39bcf8471e3ab3ef70125152c3=1701430581; _bl_uid=Rtlq0pdemvUjdwuU0mna4wefs90p; csrftoken=AZRVgBFAYbZ8j6ud6Er7hfSUYBoaMMiZyImW25xy9HYJ7Kluvhok6RKzjAofXb3H; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiMTgwMDQyMyIsIl9hdXRoX3VzZXJfYmFja2VuZCI6ImRqYW5nby5jb250cmliLmF1dGguYmFja2VuZHMuTW9kZWxCYWNrZW5kIiwiX2F1dGhfdXNlcl9oYXNoIjoiYjg1YTc3ZmYxMDgxNTAwMzg3NzY1YzE3ZGQ0M2I5YTEyYTNhOWYxNWI4YmRhNDVjNjc1ZjFiYmExNGU1YzFmNyIsImlkIjoxODAwNDIzLCJlbWFpbCI6ImR1bWluZ3poZUAxMjYuY29tIiwidXNlcm5hbWUiOiJlbGZzb25nLXYiLCJ1c2VyX3NsdWciOiJlbGZzb25nLXYiLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jbi9hbGl5dW4tbGMtdXBsb2FkL3VzZXJzL2VsZnNvbmctdi9hdmF0YXJfMTYwMDU0MTMyNy5wbmciLCJwaG9uZV92ZXJpZmllZCI6dHJ1ZSwiX3RpbWVzdGFtcCI6MTcwMTQzMDYzNy4yMjI4ODYsImV4cGlyZWRfdGltZV8iOjE3MDM5NjI4MDAsInZlcnNpb25fa2V5XyI6Mn0.xj1NZG6DUHjo12TmwApzDnzbTIS33WSfKWBAy_UkCJg; a2873925c34ecbd2_gr_last_sent_sid_with_cs1=5c028fe5-e819-4177-bc61-685d662a8ea3; a2873925c34ecbd2_gr_last_sent_cs1=elfsong-v; _ga=GA1.1.695723023.1701430580; Hm_lpvt_f0faad39bcf8471e3ab3ef70125152c3=1701430639; a2873925c34ecbd2_gr_cs1=elfsong-v; _ga_PDVPZYN3CW=GS1.1.1701430580.1.1.1701431322.52.0.0; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.e30.KpufdHIo8CeGduwC5DCQoba8bmWCjJ9mUTYQ4npFdlk; csrftoken=AZRVgBFAYbZ8j6ud6Er7hfSUYBoaMMiZyImW25xy9HYJ7Kluvhok6RKzjAofXb3H',
            'referer': 'https://leetcode.cn/problems/two-sum/submissions/486127358/',
            'sec-ch-ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'x-csrftoken': 'AZRVgBFAYbZ8j6ud6Er7hfSUYBoaMMiZyImW25xy9HYJ7Kluvhok6RKzjAofXb3H'
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        questions = response.json()
        question_dict = dict()
        for question in questions["stat_status_pairs"]:
            question_dict[question["stat"]["question__title_slug"]] = question
            
        return question_dict
    
    @staticmethod
    def remove_ansi(text):
        reaesc = re.compile(r'\x1b[^m]*m')
        new_text = reaesc.sub('', text)
        return new_text

    def execute(self, slug_name, solution, timeout=10):
        # Get question id
        question_id = self.leetcode_map[slug_name]['stat']['frontend_question_id']
        
        # Copy the solution to leetcode dir
        with open(f"{self.leetcode_code_dir}/{question_id}.{slug_name}.py", "w+") as solution_f:
            solution_f.write(solution)
            
        # Submit the solution
        with subprocess.Popen(['leetcode', 'exec', str(question_id)], stdout=subprocess.PIPE, encoding='utf-8') as process:
            process.wait(timeout=timeout)  # Returns only code: 0
            outs, errs = process.communicate(timeout=timeout)
            
        # Parse Response
        outs = OnlineJudge.remove_ansi(outs)
        outs = outs.split("\n")        
        if "on the run" in outs[1]:
            outs = outs[4:]
        
        result = {
            'status': outs[1],
            'runtime': None,
            'runtime_precentage': None,
            'memory': None,
            'memory_precentage': None,
            
        }

        if "Success" in result['status']:
            rs, ms = outs[3].split(" "), outs[5].split(" ")
            result['runtime'] = rs[1]
            result['runtime_precentage'] = rs[5]
            result['memory'] = ms[2]
            result['memory_precentage'] = ms[6]
                    
        return result
    
    def test(self, slug_name, test_cases, timeout=10):
        # Get question id
        question_id = self.leetcode_map[slug_name]['stat']['frontend_question_id']
        
        # Copy the test cases to leetcode dir
        with open(f"{self.leetcode_code_dir}/{question_id}.{slug_name}.tests.dat", "a") as case_f:
            for line in test_cases:
                case_f.write(str(line) + '\n')
            
        # Test these cases
        with subprocess.Popen(['leetcode', 'test', str(question_id)], stdout=subprocess.PIPE, encoding='utf-8') as process:
            process.wait(timeout=timeout)  # Returns only code: 0
            outs, errs = process.communicate(timeout=timeout)
            
        return outs, errs