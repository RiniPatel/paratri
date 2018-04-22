#!/usr/bin/python
import numpy as np

class CSRMatrix:
    def __init__(self, numRows, numCols):
        self.value = []
        self.IA = np.zeros(numRows + 1, np.int)#[0] * (numRows + 1)
        self.JA = []
        self.numRows = numRows
        self.numCols = numCols
    def __getitem__(self, coord):
        x, y = coord
        previous_row_values_count = self.IA[x]
        current_row_valid_count = self.IA[x+1]
        for i in range(previous_row_values_count, current_row_valid_count):
            if self.JA[i] == y:
                return self.value[i]
            else:
                return 0.0
    def __setitem__(self, coord, v):
        x, y = coord
        self.IA[x+1: self.numRows+1] += 1
        previous_row_values_count = self.IA[x]
        inserted = False
        for j in range(previous_row_values_count, self.IA[x+1]-1):
            if self.JA[j] > y:
                self.JA.insert(j, y)
                self.value.insert(j, v)
                inserted = True
                break
            elif self.JA[j] == y:
                inserted = True
                self.value[j] = v
                break
        if not inserted:
            self.JA.insert(self.IA[x+1]-1,y)
            self.value.insert(self.IA[x+1]-1, v)
    def iterate(self):
        result = [] # a list of triple (row, col, value)
        for i,v in enumerate(self.IA):
            if i == 0:
                continue
            current_row_index = 0
            while current_row_index < v-self.IA[i-1]:
                row_value = i - 1
                col_value = self.JA[self.IA[i-1] + current_row_index]
                real_value = self.value[self.IA[i-1] + current_row_index]
                result.append((row_value, col_value, real_value))
                current_row_index += 1
        return result

    def __repr__(self):
        return ('value '+ str(self.value) +
                '\nIA ' + str(self.IA) +
                '\nJA '+ str(self.JA))
