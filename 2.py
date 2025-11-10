from mrjob.job import MRJob
from datetime import datetime

class LogAna(MRJob):
    def mapper(self, _,line):
        parts=line.strip().split()
        if len(parts)==3:
            user,time,action = parts
            yield user,f"{time}\t{action}"


    def reducer(self, user, records):
        login_time=None
        total_seconds=0

        for record in records:
            time_str,action=record.split()
            time=datetime.fromisoformat(time_str)

            if action=="login":
                login_time=time 
            elif action=="logout" and login_time:
                total_seconds+=(time-login_time).total_seconds()
                login_time=None
        yield user, round(total_seconds / 3600, 2)

if __name__ == "__main__":
    LogAna.run()