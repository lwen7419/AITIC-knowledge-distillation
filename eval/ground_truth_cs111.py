PDF_PATH = "data/cs111_coursepack.pdf"

QA_PAIRS = [
    {
        "query": "What university offers CS 111 and what is the course's full name?",
        "reference": "Boston University offers CS 111, formally titled 'Introduction to Computer Science I' (CAS CS 111).",
    },
    {
        "query": "How many major units does CS 111 cover and what are they?",
        "reference": "Five major units: weeks 0-4 cover computational problem solving and functional programming; weeks 4-6 cover a look under the hood (digital logic, circuits); weeks 6-8 cover imperative programming.",
    },
    {
        "query": "What teaching approach are lectures in CS 111 based on, and who developed it?",
        "reference": "Lectures are based on peer instruction, developed by Eric Mazur at Harvard.",
    },
    {
        "query": "How many hours of work per week should students allow for CS 111?",
        "reference": "Students should allow 10–15 hours of work per week.",
    },
    {
        "query": "What are the two deadline rules for CS 111 assignments?",
        "reference": "Students can submit up to 24 hours late with a 10% penalty. No submissions are accepted after 24 hours.",
    },
    {
        "query": "What is required to earn full participation credit in CS 111?",
        "reference": "Full credit requires earning 85% of points for pre-lecture and in-lecture questions, making 85% of the lecture-attendance votes, and attending 85% of the labs.",
    },
    {
        "query": "Who is the instructor for CS 111 A1/A2 lectures?",
        "reference": "The instructor is David Sullivan.",
    },
    {
        "query": "What does the % operator do in Python?",
        "reference": "The % operator is the modulus operator; it gives the remainder of a division.",
    },
    {
        "query": "What does the / operator always produce in Python, regardless of the operands?",
        "reference": "The / operator always produces a float result.",
    },
    {
        "query": "What does the // operator do in Python and does it round?",
        "reference": "The // operator performs integer division and discards any fractional part of the result. It does not round.",
    },
    {
        "query": "What are the two types of homework problems in CS 111?",
        "reference": "Individual-only problems, which must be completed alone, and pair-optional problems, which can be completed alone or with one other student.",
    },
    {
        "query": "What is a local variable in Python according to CS 111?",
        "reference": "When a value is assigned to a variable inside a function, it creates a local variable. It belongs to that function and cannot be accessed outside of it. Function parameters are also local to that function.",
    },
    {
        "query": "What is a global variable and can it be used inside a function?",
        "reference": "A global variable is assigned outside of a function and belongs to the global scope. It can be used anywhere in the program, including inside functions, but using global variables inside functions is not recommended.",
    },
    {
        "query": "Where are variables stored in memory according to CS 111?",
        "reference": "Variables are stored in blocks of memory called frames, which are stored in a region of memory called the stack. Each function call gets its own frame for local variables, which goes away when the function returns.",
    },
    {
        "query": "What version of Python does CS 111 use?",
        "reference": "CS 111 uses Python 3, not Python 2.",
    },
    {
        "query": "What is a string in Python?",
        "reference": "A string is a sequence of characters/symbols surrounded by single or double quotes.",
    },
    {
        "query": "What is the syntax for a list comprehension with a filter condition in Python?",
        "reference": "[expression for variable in sequence if boolean]",
    },
    {
        "query": "What is a common mistake when using 'and' or 'or' in Python conditionals?",
        "reference": "Both sides of the and/or operator should be boolean expressions that could stand on their own. Writing something like 'a == 0 or 1' is problematic because '1' is an integer, not a boolean expression.",
    },
    {
        "query": "What does range(low, high) produce in Python?",
        "reference": "range(low, high) produces the range of integers from low to high-1. If low is omitted, the range starts at 0.",
    },
    {
        "query": "What are the two steps taken when an assignment statement executes in Python?",
        "reference": "Step 1: evaluate the expression on the right-hand side of the =. Step 2: assign the resulting value to the variable on the left-hand side of the =.",
    },
]
