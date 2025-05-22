#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

struct Node {
    int value;
    Node *next;
    Node *prev;
};

int main(int argc, char* argv[]) {
    srand(time(nullptr)); // Seed for random number generation

    Node *head = new Node{0, nullptr, nullptr}; // Create the first node (head)
    Node *p, *prev = nullptr; // Working pointers
    int value;
    int k;
    Node *newNode;

    int max_power = stoi(argv[1]); // Adjust this to control max nodes (e.g., 2^10)
    int num_nodes = max_power;
    auto start = chrono::high_resolution_clock::now(); // Start time
    for (int i = 1; i < num_nodes; i++) {
            value = rand() % num_nodes + 1; // ✅ Random value between 1 and num_nodes

            // Create a new node
            newNode = new Node{value, nullptr, nullptr};

            // Insert in sorted order
            p = head;
            prev = nullptr;

            while (p != nullptr && p->value < value) {
                prev = p;
                p = p->next;
            }

            if (prev == nullptr) { // Insert at head (shouldn't happen since head is always 0)
                newNode->next = head;
                head->prev = newNode;
                head = newNode;
            } else {
                newNode->next = prev->next;
                newNode->prev = prev;
                if (prev->next != nullptr) {
                    prev->next->prev = newNode;
                }
                prev->next = newNode;
            }
        }

    // Free allocated memory
    p = head;
    while (p != nullptr){ 
        Node *temp = p;
        p = p->next;
        delete temp;
    }
    auto end = chrono::high_resolution_clock::now(); // End time
    chrono::duration<double> time_taken = end - start; // Time in seconds
    cout << "Number of nodes = " << num_nodes << ", Time taken: " << time_taken.count() << " seconds" << endl;
    return 0;
}
