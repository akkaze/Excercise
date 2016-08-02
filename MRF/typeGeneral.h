#ifndef __TYPEGENERAL_H__
#define __TYPEGENERAL_H__

#include <string.h>
#include <assert.h>


template <class T> class MRFEnergy;

class TypeGeneral
{
private:
    struct Vector;
    struct Edge;

public:
    typedef enum
    {
        GENERAL,
        POTTS
    } Type;

    typedef int Label;
    typedef double REAL;
    struct GlobalSize; 
    struct LocalSize; 
    struct NodeData; 
    struct EdgeData; 

    struct GlobalSize
    {
    };

    struct LocalSize
    {
        LocalSize(int k);

    private:
        friend struct Vector;
        friend struct Edge;
        int     k_; 
    };

    struct NodeData
    {
        NodeData(Real* data); 

    private:
        friend struct Vector;
        friend struct Edge;
        Real*       data_;
    };

    struct EdgeData
    {
        EdgeData(Type type, Real lambda_potts); 
        EdgeData(Type type, Real* data); 
        private:
            friend struct Vector;
            friend struct Edge;
            Type        type_;
            union
            {
                Real    lambda_potts_;
                Real*   data_general_;
            };
    };

private:
    friend class MRFEnergy<TypeGeneral>
    struct Vector
    {
        static int getSizeInBytes(GlobalSize kglobal,LocalSize k);

    };
};