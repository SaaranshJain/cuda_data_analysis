#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <omp.h>
#include <iterator>

using namespace std;

int countWords(const string& text) {
    istringstream iss(text);
    string word;
    int count = 0;
    while (iss >> word) {
        count++;
    }
    return count;
}

vector<string> parseCSVLine(const string& line) {
    vector<string> fields;
    string field;
    bool inQuotes = false;
    
    for (size_t i = 0; i < line.length(); i++) {
        if (line[i] == '"') {
            inQuotes = !inQuotes;
        } else if (line[i] == ',' && !inQuotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field += line[i];
        }
    }
    fields.push_back(field);
    
    for (auto& f : fields) {
        if (f.size() >= 2 && f.front() == '"' && f.back() == '"') {
            f = f.substr(1, f.size() - 2);
        }
    }
    
    return fields;
}

int main() {
    ifstream file("output.csv");
    if (!file.is_open()) {
        cerr << "Error opening file" << endl;
        return 1;
    }

    vector<string> lines;
    string line;
    getline(file, line);
    
    while (getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    cout << "Total lines read: " << lines.size() << endl;

    unordered_map<string, int> elaborateReviewers;
    
    #pragma omp parallel
    {
        unordered_map<string, int> localCounts;
        
        #pragma omp for
        for (size_t i = 0; i < lines.size(); i++) {
            vector<string> fields = parseCSVLine(lines[i]);
            
            if (fields.size() >= 5) {
                string& reviewerID = fields[1];
                string& reviewText = fields[4];
                
                int wordCount = countWords(reviewText);
                if (wordCount >= 50) {
                    localCounts[reviewerID]++;
                }
            }
        }
        
        #pragma omp critical
        {
            for (const auto& [reviewerID, count] : localCounts) {
                elaborateReviewers[reviewerID] += count;
            }
        }
    }
    
    cout << "Elaborate Reviewers (5+ reviews with 50+ words):" << endl;
    for (const auto& [reviewerID, count] : elaborateReviewers) {
        if (count >= 5) {
            cout << "Reviewer ID: " << reviewerID << " - " << count << " elaborate reviews" << endl;
        }
    }
    
    cout << "Total reviewers found: " << elaborateReviewers.size() << endl;
    
    return 0;
} 