from tkinter import Entry
from tkinter import StringVar
from tkinter import Button
from tkinter import Tk
from view.callback import my_callback

def run():
    root = Tk()
    root.title("k-search")
    root.geometry("600x400+600+200")
    root.columnconfigure(0, weight = 1)
    root.columnconfigure(1, weight = 1)

    i_query = StringVar()
    input_query = Entry(root, textvariable = i_query, width = 30, bd = 3)
    input_query.grid(column = 0, row = 1, pady = 130, columnspan = 4)

    search_button = Button(root, text = "Search", command = lambda: my_callback(input_query))
    search_button.grid(column = 1, row = 1)
    root.bind("<Return>", lambda event, query=input_query: my_callback(query))
    root.mainloop()