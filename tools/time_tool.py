import time


class TimeTool:

    iniTime = 0.0
    finalTime = 0.0
    totalTime = '00h00m00s'
    endDataTime = 'Date Hour year'
    initDataTime = 'Date Hour year'

    def init(self):
        self.iniTime = time.time()
        self.initDataTime = time.ctime()

    def end(self):
        self.finalTime = time.time()
        self.endDataTime = time.ctime()
        hour = 0
        minute = 0
        second = 0
        value = self.finalTime - self.iniTime
        if value >= 3600:
            hour = int(value/3600)
            helper = value%3600
            if helper >= 60:
                minute = int(helper/60)
                second = int(helper%60)
            else:
                second = int(helper)
            self.totalTime = '{0}h:{1}m:{2}s'.format(hour, minute, second)
        elif value >= 60:
            minute = int(value/60)
            second = int(value%60)
            self.totalTime = '{0}h:{1}m:{2}s'.format(hour, minute, second)
        else:
            second = int(value)
            self.totalTime = '{0}h:{1}m:{2}s'.format(hour, minute, second)

    def getExecuTime(self):
        return self.totalTime

    def getInDateTime(self):
        return self.initDataTime

    def getEnDataTime(self):
        return self.endDataTime

