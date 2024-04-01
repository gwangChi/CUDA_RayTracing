#include <iostream>

using namespace std;

double sq(double& x){
    return x*x;
}

int main(){
    cout<<sq(5)<<endl;
}