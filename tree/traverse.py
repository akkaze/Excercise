# Python program for iterative postorder traversal
# using one stack

# Stores the answer
ans = []

# A Binary tree node
class Node:
    
    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.visited = False
def peek(stack):
    if len(stack) > 0:
        return stack[-1]
    return None
# A iterative function to do postorder traversal of 
# a given binary tree
def postOrderIterative(root):
        
    # Check for empty tree
    if root is None:
        return 

    stack = []
    
    while(True):
        
        while (root):
             # Push root's right child and then root to stack
             if root.right is not None and root.visited == False:
                stack.append(root.right)
             if root.visited == False:
                stack.append(root)

             # Set root as root's left child
             root = root.left
        
        # Pop an item from stack and set it as root
        root = stack.pop()

        # If the popped item has a right child and the
        # right child is not processed yet, then make sure
        # right child is processed before root
        if (root.right is not None and 
            peek(stack) == root.right):
            stack.pop() # Remove right child from stack 
            if root.visited == False:
                stack.append(root) # Push root back to stack
            root = root.right # change root so that the 
                             # righ childis processed next

        # Else print root's data and set root as None
        else:
            if root.right != None and root.visited == False:
                root.data = root.left.data + root.right.data
                root.visited = True
                print root.data

                # if root.visited == False:
                #     print root.data
            root = None

        if (len(stack) <= 0):
                break
def reverseLevelOrder(root):
    S = []
    Q = []
    Q.append(root)
 
    # Do something like normal level order traversal order.
    # Following are the differences with normal level order
    # traversal:
    # 1) Instead of printing a node, we push the node to stack
    # 2) Right subtree is visited before left subtree
    while(len(Q) > 0 ):
 
        # Dequeue node and make it root
        root = Q.pop(0)
        if(root.visited == False):
            S.append(root)
 
        # Enqueue right child
        if (root.right and root.right.visited == False):
            Q.append(root.right)
 
        # Enqueue left child
        if (root.left and root.left.visited == False):
            Q.append(root.left)
 
    # Now pop all items from stack one by one and print them
    while(len(S) > 0):
        root = S.pop()
        root.visited = True
        print root.data,
# Driver pogram to test above function
Nodes = []
def generate(n):
    for i in range(n + 1):
        Nodes.append(Node(i))
    for i in range(2,n + 1,1):
        Nodes[i].left = Nodes[i - 1]
        Nodes[i].right = Nodes[i - 2]
    return Nodes[n]

print "Post Order traversal of binary tree is"
root = generate(10)
reverseLevelOrder(root)
#print root.data

# This code is contributed by Nikhil Kumar Singh(nickzuck_007)
