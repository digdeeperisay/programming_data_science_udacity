# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:02:23 2019

@author: Prajamohan
"""

##### Data Types and Operators #####

print (3**2) #** is exponentiation. 9 (^ is NOT; that is bitwise XOR)
print (7//2) #3
print (-7//2) #-4

#You can check the type by using the type function

print(type(4.0))
print(type(-4))

#floating point calculations
print(.1 + .1 + .1 == .3) #False because 0.1 is little more than that. https://docs.python.org/3/tutorial/floatingpoint.html

#concise comparison operators print(san_francisco_pop_density > rio_de_janeiro_pop_density)

#string is immutable
string = 'prem is great'
print(string[1])
string[1] = 't'
#'str' object does not support item assignment

print(string*2, " ")
#('prem is greatprem is great', ' ')

#type conversion
grams = '24'
print(type(grams)) #str
grams = float(grams)
print(type(grams)) #float

#Methods (functions are similar but methods are used using the DOT notation)
#Methods are specific to the data type for a particular variable. 
#So there are some built-in methods that are available for all strings, different methods that are available for all integers, etc.

#format method is pretty useful
var = 'Great'
print('prem is {}'.format(var)) #prem is great

#some methods take additional parameters within the paranthesis
print var.isupper() #False
print var.count('a') #1

#split is important
new_str = "the cow jumped over the moon."
new_str.split()
new_str.split(' ', 3) #separator and max split. Max split + 1 is returned and the extra 1 is the remaining string

#find and rfind go from L and R respectively
print new_str.find('t') #0
print new_str.rfind('t') #20 last occurence
print(len(new_str))

########### Data Structures ###########

#### Lists ####

#lower index is inclusive and the upper index is exclusive. 
ran = [1,2,3]
print(ran[1:2]) #2
print(ran[:3]) # everything from start until index 3
print(ran[3]) # index out of range

#you can use IN and NOT IN to search for values within a list
print(1 in ran) #true

#each data structure has its own set of methods, can be ordered/unordered, mutable/unmutable

eclipse_dates = ['June 21, 2001', 'December 4, 2002', 'November 23, 2003',
                 'March 29, 2006', 'August 1, 2008', 'July 22, 2009',
                 'July 11, 2010', 'November 13, 2012', 'March 20, 2015',
                 'March 9, 2016']                 
#Modify this line so it prints the last three elements of the list
print(eclipse_dates[-3:]) #this is cool

sentence2 = ["I", "wish", "to", "register", "a", "complaint", "."]
sentence2[0:2] = ["We", "want"] #can change simultaneously
print(sentence2)

#list methods
a = [1,2,3]
b = a
a[1] = 4
print(a,b) #both changed to 4

#cant use max or > when there are mix of data types in a list

#len, max, sorted are useful list functions
x = [1,6,2,1,4,1,5,9]
print(sorted(x)) #[1, 1, 1, 2, 4, 5, 6, 9]

#join method for joining list of strings with some separator
print('-'.join(['1','2'])) #1-2; JOIN takes one argument

#append adds elements to list
k = [1,2,3]
k.append(4)
print(k) #[1, 2, 3, 4]
k.append(5)
print(k) #[1, 2, 3, 4, 5]

#### Tuples ####

#tuples are useful when things are really related like latitude, longitude

#parantheses are optional for tuples
dimensions = 1, 2, 3
print(type(dimensions)) #tuple
l, w, h = dimensions
print(l) #1
print(dimensions[1]) #2

##### Sets #####

#lists = mutable ordered
#tuple = immutable ordered
#set = mutable unordered, created with {}, no indexing and no assignment

lista = [1,1,1,1,2,3,3,4]
seta = set(lista)
print(seta) #set([1, 2, 3, 4])
seta.add(10)
seta #{1, 2, 3, 4, 10}
seta.add(10)
seta
#same

#set also has POP but a random element is removed since there is no order

#there are also UNION, INTERSECTION operators

#### Dictionaries ####

#if you define a variable with an empty set of curly braces like this: a = {},
#Python will assign an empty dictionary to that variable
 
#The keys NEED TO BE IMMUTABLE. So can't use LISTS as keys

elements = {'hydro':1, 'oxy': 2}
print(elements['hydro']) #1
print(elements.get('hydro')) #1
print(elements['xxx']) #Key error
print(elements.get('xxx')) #None So Get is better if you are not sure the key exists. You canalso retrun default value

elements['lith'] = 3
print(elements)
#{'lith': 3, 'oxy': 2, 'hydro': 1}

print('carbon' in elements) #False

#identity operators IS and IS NOT
n = elements.get("dilithium")
print(n is None) #true
print(n is not None) #false

print(elements.get('kryptonite', 'There\'s no such element!')) #"There's no such element!" DEFAULT

#IDENTITY VS EQUALITY
a = [1, 2, 3]
b = a
c = [1, 2, 3]

print(a == b)
print(a is b)
print(a == c)
print(a is c)
#TRUE TRUE TRUE FALSE

#Compound dictionaries
elements = {'hydrogen': {'number': 1, 'weight': 1.00794, 'symbol': 'H'},
            'helium': {'number': 2, 'weight': 4.002602, 'symbol': 'He'}}

elements['hydrogen']['is_noble_gas'] = False #adding noble gas key to the inner container
elements['helium']['is_noble_gas'] = True #adding noble gas key to the inner container

print(elements)
elements.keys() #prints keys

#length of dictionary can be found by using len(dictionary) since keys are unique
print(len(elements))

##### Control Flow and Conditional Statements #####

#An iterable is an object that can return one of its elements at a time. 
#This can include sequence types, such as strings, lists, and tuples, as well as non-sequence types, such as dictionaries and files.

#Here are most of the built-in objects that are considered False in Python:

#constants defined to be false: None and False
#zero of any numeric type: 0, 0.0, 0j, Decimal(0), Fraction(0, 1)
#empty sequences and collections: '"", (), [], {}, set(), range(0)

error = 1
if error:
    print 'there is an error'
    print('there is an error')
    
#basically you can use the actual values of variables to say T or F
cities = ['mumbai','bangalore']
for city in cities:
    print city.title()
#makes it caps

print(range(4))
#[0,1,2,3]
print(type(range(4)))
#list
#range(start=0,stop,step=1)

#modifying in place; use range with length function
for index in range(len(cities)):
    cities[index] = cities[index].title()
print(cities)

#converting to lower case and replacing spaces with _
names = ["Joey Tribbiani", "Monica Geller", "Chandler Bing", "Phoebe Buffay"]
usernames = []

for name in names:
    usernames.append(name.lower().replace(" ", "_"))

print(usernames)

#Counting words in a string using GET method
book_title =  ['great', 'expectations','the', 'adventures', 'of', 'sherlock','holmes','the','great','gasby','hamlet','adventures','of','huckleberry','fin']
word_counter = {}

for word in book_title:
    word_counter[word] = word_counter.get(word,0) + 1

print(word_counter)

#iterating over dictionary - just keys and keys/values
cast = {
           "Jerry Seinfeld": "Jerry Seinfeld",
           "Julia Louis-Dreyfus": "Elaine Benes",
           "Jason Alexander": "George Costanza",
           "Michael Richards": "Cosmo Kramer"
       }

for key in cast:
    print(key)

for key in cast.keys():
    print(key)

for value in cast.values():
    print(value)
        
for key, value in cast.items():
    print("Actor: {}    Role: {}".format(key, value))

# While loops
    
#factorial with while loops
number = 6
# start with our product equal to one
product = 1
# track the current number being multiplied
current = 1
while  current <= number:
    # multiply the product so far by the current number
    product *= current
    # increment current with each iteration until it reaches number
    current += 1
# print the factorial of number
print(product)

#factorial with for loops
# number we'll find the factorial of
number = 6
# start with our product equal to one
product = 1

# calculate factorial of number with a for loop
for num in range(2, number + 1):
    product *= num

# print the factorial of number
print(product)

#for loops are ideal when the number of iterations is known or finite
#while loops are ideal when the iterations need to continue until a condition is met

#Break and Continue

#Break - terminates the loop immediately
#Continue - skips one iteration

#question
#Write a loop with a break statement to create a string, news_ticker, that is exactly 140 characters long. 
#You should create the news ticker by adding headlines from the headlines list, inserting a space in
#between each headline. If necessary, truncate the last headline in the middle so that news_ticker 
#is exactly 140 characters long.

headlines = ["Local Bear Eaten by Man",
             "Legislature Announces New Laws",
             "Peasant Discovers Violence Inherent in System",
             "Cat Rescues Fireman Stuck in Tree",
             "Brave Knight Runs Away",
             "Papperbok Review: Totally Triffic"]

news_ticker = ""
for headline in headlines:
    news_ticker += headline + " "
    if len(news_ticker) >= 140:
        news_ticker = news_ticker[:140]
        break

print(news_ticker)

#quesion
#Check prime numbers
check_prime = [26, 39, 51, 53, 57, 79, 85]

# iterate through the check_prime list
for num in check_prime:
# search for factors, iterating through numbers ranging from 2 to the number itself
    for i in range(2, num):
# number is not prime if modulo is 0
        if (num % i) == 0:
            print("{} is NOT a prime number, because {} is a factor of {}".format(num, i, num))
            break
# otherwise keep checking until we've searched all possible factors, and then declare it prime
        if i == num -1:    
            print("{} IS a prime number".format(num))

#Zip and Enumerate 
            
#zip returns an iterator that combines multiple iterables into one sequence of tuples. 
#Each tuple contains the elements in that position from all the iterables.
            
print list(zip(['a', 'b', 'c'], [1, 2, 3]))
#[('a', 1), ('b', 2), ('c', 3)]

letters = ['a', 'b', 'c']
nums = [1, 2]

for letter, num in zip(letters, nums):
    print("{}: {}".format(letter, num))
#c is ignored completely

#In addition to zipping two lists together, you can also unzip a list into tuples using an asterisk.

some_list = [('a', 1), ('b', 2), ('c', 3)]
letters, nums = zip(*some_list)
print(letters,nums)

#enumerate is a built in function that returns an iterator of tuples containing indices and values of a list. You'll often use this when you want the index along with each element of an iterable in a loop.

letters = ['a', 'b', 'c', 'd', 'e']
for i, letter in enumerate(letters):
    print(i, letter)

#Question 
#Use zip to write a for loop that creates a string specifying the label and coordinates of each point and appends it to the list points.
#Each string should be formatted as label: x, y, z. For example, the string for the first coordinate should be F: 23, 677, 4.

x_coord = [23, 53, 2, -12, 95, 103, 14, -5]
y_coord = [677, 233, 405, 433, 905, 376, 432, 445]
z_coord = [4, 16, -6, -42, 3, -6, 23, -1]
labels = ["F", "J", "A", "Q", "Y", "B", "W", "X"]

points = []
for point in zip(labels, x_coord, y_coord, z_coord):
    points.append("{}: {}, {}, {}".format(*point)) #unpacking using *

for point in points:
    print(point)

#dictionary using zips
cast_names = ["Barney", "Robin", "Ted", "Lily", "Marshall"]
cast_heights = [72, 68, 72, 66, 76]

cast = dict(zip(cast_names, cast_heights))
print(cast)

#unzip tuples
cast = (("Barney", 72), ("Robin", 68), ("Ted", 72), ("Lily", 66), ("Marshall", 76))

names, heights = zip(*cast)
print(names)
print(heights)

#list comprehensions
squares = [z**2 for z in range(4)]
print squares

squares_even = [z**2 for z in range(9) if z%2 == 0]
print squares_even

#### Functions ####

def cylinder_volume(height, radius):
    pi = 3.14159
    return height * pi * radius ** 2

cylinder_volume(4,5) #314.159

#Not all functions need a return statement

def cylinder_volume_default(height, radius=5):
    pi = 3.14159
    return height * pi * radius ** 2

print cylinder_volume_default(10)  # pass in arguments by position
print cylinder_volume_default(height=10, radius=5)  # pass in arguments by name
#785.3975

def readable_timedelta(days):
    # use integer division to get the number of weeks
    weeks = days // 7
    # use % to get the number of days that remain
    remainder = days % 7
    return "{} week(s) and {} day(s).".format(weeks, remainder)

# test your function
print(readable_timedelta(20))
#2 week(s) and 6 day(s).

#Variable Scope
#Good practice: It is best to define variables in the smallest scope they will be needed in. 
#While functions can refer to variables defined in a larger scope, this is very rarely a good idea 
#since you may not know what variables you have defined if your program has a lot of variables.

#variables defined within a function cannot be used outside but things that are defined
#outside can be used within a function though this is not good practice

#question
egg_count = 0

def buy_eggs():
    egg_count += 12 # purchase a dozen eggs

buy_eggs()
#gives an error [UnboundLocalError: local variable 'egg_count' referenced before assignment]
#this is because you can't modify a variable declared outside; we can pass as argument and make changes

#lambda expressions
def multiply(x, y):
    return x * y

multiply_lambda = lambda x, y: x * y

print(multiply(3,2))
print(multiply_lambda(3,2))
#6 6
#lambda keyword; colon and arguments before colon; and expression after colon

#Map

#map() is a higher-order built-in function that takes a function and iterable as inputs, 
#and returns an iterator that applies the function to each element of the iterable

numbers = [
              [34, 63, 88, 71, 29],
              [90, 78, 51, 27, 45],
              [63, 37, 85, 46, 22],
              [51, 22, 34, 11, 18]
           ]

def mean(num_list):
    return sum(num_list) / len(num_list)

averages = list(map(mean, numbers)) #maps mean to iterable numbers
print(averages)
#[57, 58, 50, 27]

#filter() is a higher-order built-in function that takes a function and iterable as inputs
#and returns an iterator with the elements from the iterable for which the function returns True.

cities = ["New York City", "Los Angeles", "Chicago", "Mountain View", "Denver", "Boston"]

def is_short(name):
    return len(name) < 10 #returns Boolean variable

short_cities = list(filter(is_short, cities))  #filters iterable cities based on is_short
print(short_cities)

#### Scripting ####

name = input("Enter your name: ")
print("Hello there, {}!".format(name.title()))

#Try Catch

#try block; except block; else block (if no exceptions); finally block always runs
#we can specify error types in the except block

x = 0
try:
    print(0/x)
except:
    print('div by zero error')
    
#You can print the actual error message even when using except
try:
    print(0/0)
except Exception as e:
   # some code
   print("Exception occurred: {}".format(e))
#Exception occurred: integer division or modulo by zero   
#Exception is the base class for all types of errors

#Reading and Using files
   
#open function returns a file object

#close a file after we are done with it because the file handles are limited
   
f = open('my_path/my_file.txt', 'r') #r is read, w is write (it will overwrite), a is append
file_data = f.read()
f.close()

f = open('my_path/my_file.txt', 'w')
f.write("Hello there!")
f.close()

#read close write functions are applied to the file object

with open('my_path/my_file.txt', 'r') as f:
    file_data = f.read()
    
#This with keyword allows you to open a file, do operations on it, and automatically 
#close it after the indented code is executed, in this case, reading from the file. 
#Now, we don’t have to call f.close()! You can only access the file object, f, 
#within this indented block.

#example
camelot_lines = []
with open("camelot.txt") as f:
    for line in f: #identified using the /n character
        camelot_lines.append(line.strip()) #removes leading newline characters

print(camelot_lines)

#Import files or modules
#When we do 'import x' we create a module 'x' and then to access objects from it,
#we use the .dotation 

import pandas as pd
#pd.concat etc.

#There is a standard python library as well

import math
print(math.factorial(4))
#24

#random passwords
import random

word_list = ['a','b','c','d','e']

def generate_password():
    return ''.join(random.sample(word_list,3))

#random.sample(population, k)
#Return a k length list of unique elements chosen from the population sequence. 
#Used for random sampling without replacement.
    
print(generate_password())
#abd
#dae

#Different import statements

#import full module - IMPORT X
#import specific object within module - FROM X IMPORT Y
#import and rename - IMPORT X AS Y
#package - module that contains submodules - IMPORT packageName.submodule_name

#### Numpy ####

#NumPy stands for Numerical Python and it's a fundamental package for scientific computing in Python. 
#Pandas is built on top of Numpy

#much more faster and efficient than python lists etc. They are memory efficient and optimized
#can work with matrices and vectors etc.

import time
import numpy as np
x = np.random.random(100000)

start = time.time()
print sum(x)/len(x)
print(time.time() - start)

start = time.time()
print(np.mean(x))
print(time.time() - start)

#numpy is much faster

#At the core of NumPy is the ndarray, where nd stands for n-dimensional. 
#An ndarray is a multidimensional array of elements all of the same type. 
#In other words, an ndarray is a grid that can take on many shapes and can hold either numbers or strings. 

## Creating arrays from Python lists ##

#We create a 1D ndarray that contains only integers
x = np.array([1, 2, 3, 4, 5]) # just a function that returns an ndarray

# Let's print the ndarray we just created using the print() command
print('x = ', x)

print('x has dimensions:', x.shape)
print('x is an object of type:', type(x))
print('The elements in x are of type:', x.dtype)
#('x has dimensions:', (5L,))
#('x is an object of type:', <type 'numpy.ndarray'>)
#('The elements in x are of type:', dtype('int32'))

#2D arrays
# We create a rank 2 ndarray that only contains integers
Y = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]) #each is a row

# We print Y
print()
print('Y = \n', Y)
print()

# We print information about Y
print('Y has dimensions:', Y.shape) #row, columns
print('Y has a total of', Y.size, 'elements')
print('Y is an object of type:', type(Y))
print('The elements in Y are of type:', Y.dtype)
#shape, size, typ, dtype are useful functions to know about an N-D arrays

#If numpy sees different data types in the nd-array, it will UPCAST it. I.E int to flaot
#but you can also specify the dtype like below

x = np.array([1.5, 2.2, 3.7, 4.0, 5.9], dtype = np.int64)
print(type(x))
print(x.dtype) #int64

#You can save and load numpy arrays using NP.SAVE and NP.LOAD functions!

## Creating arrays using built-in functions ##

#np.zeroes
z = np.zeros(4)
print(z) #[0. 0. 0. 0.]

X = np.zeros((3,4))
print(X)
#[[0. 0. 0. 0.]
# [0. 0. 0. 0.]
# [0. 0. 0. 0.]]

#np.ones
o = np.ones(4)
print(o) #[1. 1. 1. 1.]
print(o.dtype) #float64

o = np.ones(4, dtype = int)
print(o) #[1. 1. 1. 1.]
print(o.dtype) #int32

#there in np.ones, np.full, np.eye (identity matrix)
# We create a 5 x 5 Identity matrix. 
X = np.eye(5)

# We print X
print()
print('X = \n', X)
print()

# We print information about X
print('X has dimensions:', X.shape)
print('X is an object of type:', type(X))
print('The elements in X are of type:', X.dtype) 
#('X = \n', array([[1., 0., 0., 0., 0.],
#       [0., 1., 0., 0., 0.],
#       [0., 0., 1., 0., 0.],
#       [0., 0., 0., 1., 0.],
#       [0., 0., 0., 0., 1.]]))

# Create a 4 x 4 diagonal matrix that contains the numbers 10,20,30, and 50
# on its main diagonal
X = np.diag([10,20,30,50])

# We print X
print()
print('X = \n', X)
print()

#np.arange is pretty important: Array of evenly spaced values.
#arange([start,] stop[, step,][, dtype])

print(np.arange(5)) #[0 1 2 3 4]
print(np.arange(5,10)) #[5 6 7 8 9]

#np.reshape
# We create a rank 1 ndarray with sequential integers from 0 to 19
x = np.arange(20)

# We print x
print()
print('Original x = ', x)
print()

# We reshape x into a 4 x 5 ndarray 
x = np.reshape(x, (4,5))
print('Modified x = ', x)
#('Original x = ', array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#       17, 18, 19]))
#()
#('Modified x = ', array([[ 0,  1,  2,  3,  4],
#       [ 5,  6,  7,  8,  9],
#       [10, 11, 12, 13, 14],
#       [15, 16, 17, 18, 19]]))

x = np.reshape(x, (4,3))
#ValueError: cannot reshape array of size 20 into shape (4,3)

#linspace is important too
#numpy.linspace(start, stop, num = 50, endpoint = True, retstep = False, dtype = None) : Returns number spaces evenly w.r.t interval. Similar to arange but instead of step it uses sample number.

# restep set to True 
print("B\n", np.linspace(2.0, 3.0, num=5, retstep=True), "\n") 
  
# To evaluate sin() in long range  
x = np.linspace(0, 2, 10) 
print("A\n", np.sin(x)) 

print(np.linspace(1,10,10))
#[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]

print(np.linspace(1,10,1))
#[ 1.]

##Accessing elements, slicing and deleting

x = np.arange(1,10).reshape(3,3)
print(x)
print(x[1][1]) #5

#np.delete to delete elements; for 2+d arrays, need to specify axis. axis= 0 rows; axis = 1 columns
print(x)
y = np.delete(x,1,axis=1)
print(x)
print(y)
#[[1 2 3]
# [4 5 6]
# [7 8 9]]
#[[1 3]
# [4 6]
# [7 9]]

#np.append(ndarray, elements, axis)
#np.insert(ndarray, index, elements, axis)

#Slicing
#1. ndarray[start:end]
#2. ndarray[start:]
#3. ndarray[:end]

#We say that slicing only creates a view of the original array. 
#This means that if you make changes in Z you will be in effect changing the elements in X as well. Let's see this with an example:

#np.copy(ndarray) to actually copy and not make changes to actual array

#Other functions
# We sort x but only keep the unique elements in x
print(x)
print(np.sort(np.unique(x)))
#[[1 2 3]
# [4 5 6]
# [7 8 9]]
#[1 2 3 4 5 6 7 8 9]

#Boolean indexing
print(x[x>5])
#[6 7 8 9]

#numpy allows element-wise operations as well as matrix operations
#np.add, np.subtract etc.
#must have same shape or be broadcastable

# We create two rank 1 ndarrays
x = np.array([1,2,3,4])
y = np.array([5.5,6.5,7.5,8.5])

# We print x
print('x = ', x)

# We print y
print('y = ', y)

#('x = ', array([1, 2, 3, 4]))
#('y = ', array([5.5, 6.5, 7.5, 8.5]))

# We perfrom basic element-wise operations using arithmetic symbols and functions
print('x + y = ', x + y)
#array([ 6.5,  8.5, 10.5, 12.5]))
print('add(x,y) = ', np.add(x,y))
print()
print('x - y = ', x - y)
print('subtract(x,y) = ', np.subtract(x,y))
print()
print('x * y = ', x * y)
print('multiply(x,y) = ', np.multiply(x,y))
print()
print('x / y = ', x / y)
print('divide(x,y) = ', np.divide(x,y))

#broadcasting
#The term broadcasting describes how numpy treats arrays with different shapes during 
#arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” 
#across the larger array so that they have compatible shapes. 
#Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python

##### Pandas ######

# Panel Data: Pandas incorporates two additional data structures into Python, namely Pandas Series and Pandas DataFrame. 
#These data structures allow us to work with labeled and relational data in an easy and intuitive manner

#Benefits of pandas
#Allows the use of labels for rows and columns
#Can calculate rolling statistics on time series data
#Easy handling of NaN values
#Is able to load data of different formats into DataFrames
#Can join and merge different datasets together
#It integrates with NumPy and Matplotlib

#SERIES (can hold multiple data types)

import pandas as pd
groceries = pd.Series(data = [30,6,'yes','no'], index = ['egg','apple','milk','bread'])
print(groceries)
#egg       30
#apple      6
#milk     yes
#bread     no
#dtype: object
print(groceries['egg'])
#30
print(groceries[0])
#30

groceries = pd.Series(data = [30,6,'yes','no'], index = ['egg','milk','bread'])
print(groceries)
#error -  legnth should match

#important attributes

print(groceries.shape, groceries.size, groceries.index, groceries.values)

#Pandas Series have two attributes, .loc and .iloc to explicitly state what we mean. 
#The attribute .loc stands for location and it is used to explicitly state that we are using a labeled index. 
#Similarly, the attribute .iloc stands for integer location and it is used to explicitly state that we are using a numerical index.

# We access elements in Groceries using index labels:

# We use a single index label
print('How many eggs do we need to buy:', groceries['eggs'])
print()

# we can access multiple index labels
print('Do we need milk and bread:\n', groceries[['milk', 'bread']]) 
print()

# we use loc to access multiple index labels
print('How many eggs and apples do we need to buy:\n', groceries.loc[['eggs', 'apples']]) 
print()

# We access elements in Groceries using numerical indices:

# we use multiple numerical indices
print('How many eggs and apples do we need to buy:\n',  groceries[[0, 1]]) 
print()

# We use a negative numerical index
print('Do we need bread:\n', groceries[[-1]]) 
print()

# We use a single numerical index
print('How many eggs do we need to buy:', groceries[0]) 
print()
# we use iloc to access multiple numerical indices
print('Do we need milk and bread:\n', groceries.iloc[[2, 3]]) 

#drop does NOT affect original series. We can change this by giving inplace = True
# We display the original grocery list
print('Original Grocery List:\n', groceries)

# We remove apples from our grocery list. The drop function removes elements out of place
print()
print('We remove apples (out of place):\n', groceries.drop('egg'))

# When we remove elements out of place the original Series remains intact. To see this
# we display our grocery list again
print()
print('Grocery List after removing apples out of place:\n', groceries)

#('Original Grocery List:\n', egg       30
#apple      6
#milk     yes
#bread     no
#dtype: object)
#()
#('We remove apples (out of place):\n', apple      6
#milk     yes
#bread     no
#dtype: object)
#()
#('Grocery List after removing apples out of place:\n', egg       30
#apple      6
#milk     yes
#bread     no
#dtype: object)

# We remove apples from our grocery list in place by setting the inplace keyword to True
groceries.drop('egg', inplace = True) #for series, we don't need an axis. We drop using index
print(groceries)

#apple      6
#milk     yes
#bread     no
#dtype: object

groceries.iloc[0:3]
#slicing using iloc if index is non-numeric; else below works

dummy = pd.Series([1,2,3,4])
print(dummy)
dummy[0:3]

#Arithmetic operators

#We can do element wise operations here as well similar to numpy
# We create a Pandas Series that stores a grocery list of just fruits
fruits= pd.Series(data = [10, 6, 3,], index = ['apples', 'oranges', 'bananas'])

# We display the fruits Pandas Series
fruits

# We print fruits for reference
print('Original grocery list of fruits:\n ', fruits)

# We perform basic element-wise operations using arithmetic symbols
print()
print('fruits + 2:\n', fruits + 2) # We add 2 to each item in fruits
print()
print('fruits - 2:\n', fruits - 2) # We subtract 2 to each item in fruits
print()
print('fruits * 2:\n', fruits * 2) # We multiply each item in fruits by 2 
print()
print('fruits / 2:\n', fruits / 2) # We divide each item in fruits by 2
print()

#We can also use numpy functions
# We import NumPy as np to be able to use the mathematical functions
# We print fruits for reference
print('Original grocery list of fruits:\n', fruits)

# We apply different mathematical functions to all elements of fruits
print()
print('EXP(X) = \n', np.exp(fruits))
print() 
print('SQRT(X) =\n', np.sqrt(fruits))
print()
print('POW(X,2) =\n',np.power(fruits,2)) # We raise all elements of fruits to the power of 2

#You can apply the operations on specific items as well using index
# We print fruits for reference
print('Original grocery list of fruits:\n ', fruits)
print()

# We add 2 only to the bananas
print('Amount of bananas + 2 = ', fruits['bananas'] + 2)
print()

# We subtract 2 from apples
print('Amount of apples - 2 = ', fruits.iloc[0] - 2)
print()

# We multiply apples and oranges by 2
print('We double the amount of apples and oranges:\n', fruits[['apples', 'oranges']] * 2)
print()

# We divide apples and oranges by 2
print('We half the amount of apples and oranges:\n', fruits.loc[['apples', 'oranges']] / 2)

#If applying to entire series, the arithmetic operator should be valid for all elements and hence all data types

#Data Frames

# We import Pandas as pd into Python
import pandas as pd

# We create a dictionary of Pandas Series 
items = {'Bob' : pd.Series(data = [245, 25, 55], index = ['bike', 'pants', 'watch']),
         'Alice' : pd.Series(data = [40, 110, 500, 45], index = ['book', 'glasses', 'bike', 'pants'])}

# We print the type of items to see that it is a dictionary
print(type(items))
# We create a Pandas DataFrame by passing it a dictionary of Pandas Series
shopping_carts = pd.DataFrame(items) #key are columns and values are values

# We display the DataFrame
shopping_carts
#         Alice    Bob
#bike     500.0  245.0
#book      40.0    NaN
#glasses  110.0    NaN
#pants     45.0   25.0
#watch      NaN   55.0

#the indices are unioned
#So whenever a DataFrame is created, if a particular column doesn't have values for a
#particular row index, Pandas will put a NaN value there

#If we don't provide index labels to the Pandas Series, Pandas will use numerical row indexes when it creates the DataFrame. Let's see an example:

# We create a dictionary of Pandas Series without indexes
data = {'Bob' : pd.Series([245, 25, 55]),
        'Alice' : pd.Series([40, 110, 500, 45])}

# We create a DataFrame
df = pd.DataFrame(data)

# We display the DataFrame
df
#   Alice    Bob
#0     40  245.0
#1    110   25.0
#2    500   55.0
#3     45    NaN

#We can subset the columns and the records using COLUMNS and INDEX filters as shown below
#We can combine this too. i..e use both COLUMNS and INDEX filter together

# We Create a DataFrame that only has Bob's data
bob_shopping_cart = pd.DataFrame(items, columns=['Bob']) #if you give a random column here, it will make it NaN

# We display bob_shopping_cart
bob_shopping_cart

# We Create a DataFrame that only has selected items for both Alice and Bob
sel_shopping_cart = pd.DataFrame(items, index = ['pants', 'book']) #if you give a random index here, it will make it NaN

# We display sel_shopping_cart
sel_shopping_cart

#accessing elements
#It is important to know that when accessing individual elements in a DataFrame, 
#as we did in the last example above, the labels should always be provided with the 
#column label first, i.e. in the form dataframe[column][row]

# We create a list of Python dictionaries
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35}, 
          {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]

# We create a DataFrame  and provide the row index
store_items = pd.DataFrame(items2, index = ['store 1', 'store 2'])

# We display the DataFrame
store_items
#         bikes  glasses  pants  watches
#store 1     20      NaN     30       35
#store 2     15     50.0      5       10

# We access rows, columns and elements using labels
print()
print('How many bikes are in each store:\n', store_items[['bikes']])
print()
print('How many bikes and pants are in each store:\n', store_items[['bikes', 'pants']])
print()
print('What items are in Store 1:\n', store_items.loc[['store 1']])
print()
print('How many bikes are in Store 2:', store_items['bikes']['store 2'])
print()
print('How many bikes are in Store 2:', store_items['store 2']['bikes']) #error since row comes before column

#Adding a new column
store_items['shirts'] = [15,2]

print(store_items)
#         bikes  glasses  pants  watches  shirts
#store 1     20      NaN     30       35      15
#store 2     15     50.0      5       10       2

#Useful functions

#df.append store_items = store_items.append(new_store); adds a new row
#df.insert dataframe.insert(loc,label,data)
#df.pop

#dropping columns
# We remove the watches and shoes columns
store_items = store_items.drop(['watches', 'bikes'], axis = 1) #xis =0 is for rows

# we display the modified DataFrame
store_items
#         glasses  pants  shirts
#store 1      NaN     30      15
#store 2     50.0      5       2

#Nan values

print store_items.isnull()
#         glasses  pants  shirts
#store 1     True  False   False
#store 2    False  False   False

print store_items.isnull().sum()
#glasses    1
#pants      0
#shirts     0
#dtype: int64

print store_items.isnull().sum().sum()
#1 We sum twice because the first SUM returns a series with T F values i.e. 1 0 values

# We drop any rows with NaN values
store_items.dropna(axis = 0)
# We drop any columns with NaN values
store_items.dropna(axis = 1)

#         pants  shirts
#store 1     30      15
#store 2      5       2

#Use inplace = True to make changes directly to the dataframe; else they are not removed

#Use fillna to fill missing values with something else
store_items.fillna(0)

#fillna has some functions like ffill, backfill that can propogate values on certain axes

#useful to check for null values

#Google_stock.isnull().any()
#Date                  False
#Open                False
#High                  False
#Low                   False
#Close                 False
#Adj Close          False
#Volume             False
#dtype: bool

# head(), tail(), describe() [can do this for a single column as well] are super useful
#mean(), min(), max(), corr() etc. are also useful
#group by also super useful

data.groupby(['Name'])['Salary'].sum()
#
#Iterators And Generators
#Iterables are objects that can return one of their elements at a time, such as a list. Many of the built-in functions we’ve used so far, like 'enumerate,' return an iterator.

#An iterator is an object that represents a stream of data. This is different from a list, which is also an iterable, but is not an iterator because it is not a stream of data.

#Generators are a simple way to create iterators using functions. You can also define iterators using classes, which you can read more about here.

#Here is an example of a generator function called my_range, which produces an iterator that is a stream of numbers from 0 to (x - 1).

def my_range(x):
    i = 0
    while i < x:
        yield i
        i += 1
#Notice that instead of using the return keyword, it uses yield. This allows the function to return values one at a time, and start where it left off each time it’s called. This yield keyword is what differentiates a generator from a typical function.

#The yield statement suspends function’s execution and sends a value back to caller, 
#but retains enough state to enable function to resume where it is left off. 
#When resumed, the function continues execution immediately after the last yield run. This allows its code to produce a series of values over time, rather them computing them at once and sending them back like a list.

sq_list = [pop**2 for pop in range(10)]  # this produces a list of squares

sq_iterator = (pop**2 for pop in range(10))  # this produces an iterator of squares

######## Project ########

x = raw_input("Enter a name")
if x == "Prem":
    print(x)
    print("Prem is here")
else:
    print(x)
    print("Someone other than Prem")































































