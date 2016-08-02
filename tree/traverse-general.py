class Node:
    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.childs = []
        self.visited = False
def peek(stack):
    if len(stack) > 0:
        return stack[-1]
    return None

class Fib:
    def __init__(self, data):
        self.data = data
        self.childs = []
        self.visited = False

def fibonacci(n):
    stack = []
    stack.append(Fib(1))
    fib = 0
    while(len(stack) != 0 ):
        num = stack.pop()
        fib = num 
        if(num == n ):
                stack.append(node.childs[i])
        else:
            continue

def preorder(root):
    stack = []
    stack.append(root)
    node = None

    while(len(stack) != 0 ):
        node = stack.pop()
        print node.data

        if(len(node.childs) !=0 ):
            for i in range(len(node.childs) - 1,-1,-1):
                stack.append(node.childs[i])
        else:
            continue

def postorder(root):
    stack = []
    stack.append(root)
    node = None
    last_node = None

    while(len(stack) != 0):
        node = stack.pop()
        #print node.data
        if(len(node.childs) != 0 and last_node != node.childs[0]):

            for i in range(len(node.childs) - 1,-1,-1):
                stack.append(node.childs[i])
            # print len(stack)
        else:
            print node.data
            last_node = node
            #stack.pop()
            # print len(stack)


def postOrderIterative(root):
     
    # Create two stacks 
    s1 = []
    s2 = []
     
    # Push root to first stack
    s1.append(root)
     
    # Run while first stack is not empty
    while(len(s1) >0):
         
        # Pop an item from s1 and append it to s2
        node = s1.pop()
        s2.append(node)
     
        # Push left and right children of removed item to s1
        if(len(node.childs) != 0):

            for i in range(len(node.childs)):
                s1.append(node.childs[i])
        else:
            continue
        # Print all eleements of second stack
    while(len(s2) > 0):
        node = s2.pop()
        print node.data,
 
def postorderRecursive(root):
    if(len(root.childs) != 0):
        for i in range(len(root.childs)):
            postorderRecursive(root.childs[i])
        print root.data,
    else:
        print root.data,
        return
root = Node(10)
root.childs = [Node(8),Node(2)]
root.childs[0].childs = [Node(3),Node(5),Node(4)]
root.childs[1].childs = [Node(1),Node(11),Node(12),Node(6)]
postorderRecursive(root)
print '\n'
postOrderIterative(root)
