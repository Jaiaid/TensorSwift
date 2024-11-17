#ifndef _DATA_H
#define _DATA_H

class Data
{
    int num;
public:
    Data(){ num=0; }
    Data(int num){ this->num=num; }

    int get()const { return this->num; }
    Data operator+(const Data& d)const { return Data(this->num + d.get()); }
    Data operator-(const Data& d)const { return Data(this->num - d.get()); }
    Data operator*(const Data& d)const { return Data(this->num * d.get()); }
    Data operator/(const Data& d)const { return Data(this->num / d.get()); }

    friend Data operator+(int n, const Data& d) { return Data(n + d.num); };
    friend Data operator-(int n, const Data& d) { return Data(n - d.num); };
    friend Data operator*(int n, const Data& d) { return Data(n * d.num); };
    friend Data operator/(int n, const Data& d) { return Data(n / d.num); };
};

#endif