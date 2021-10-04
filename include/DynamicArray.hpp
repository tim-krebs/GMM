#pragma once 

#include <cstddef>
#include <utility>

template <typename T>
class DynamicArray
{
    
private:
    std::size_t m_length;
    std::size_t m_capacity;
    T *m_data;
    
public:
    // Constructor/Destructor
    DynamicArray();
    DynamicArray(const T &value, const std::size_t length);
    ~DynamicArray() noexcept;

    //Copy Constructor/Assignment Operator
    DynamicArray(const DynamicArray &other);
    DynamicArray& operator=(const DynamicArray &other);

    // Move-Constructor/ Assignment Operator
    DynamicArray(DynamicArray &&other) noexcept;
    DynamicArray& operator=(DynamicArray &&other) noexcept;

    //Data manipulation methods
    void push_back(const T &value);
    void pop_back();

    //Operation overloading
    T &operator[](const std::size_t index);
    const T &operator[](const std::size_t index) const;

    // Helper methods
    std::size_t get_length() const;

};

/**
 * @brief Construct a new Dynamic Array< T>:: Dynamic Array object
 * 
 * @tparam T 
 */
template <typename T>
DynamicArray<T>::DynamicArray(): m_length(0), m_capacity(0), m_data(nullptr)
{

}

/**
 * @brief Construct a new Dynamic Array< T>:: Dynamic Array object
 * 
 * @tparam T 
 * @param value  The fill vlaue of the array
 * @param length The length of the array
 */
template <typename T>
DynamicArray<T>::DynamicArray(const T &value, const std::size_t length): m_length(length), m_capacity(length), m_data(new T[length])
{
    for(std::size_t i = 0; i < length; ++i)
    {
        m_data[i] = value;
    }
}

/**
 * @brief Destroys a dynamic array object
 */
template <typename T>
DynamicArray<T>::~DynamicArray() noexcept
{
    if (m_data != nullptr)
    {
        delete[] m_data;
        m_data = nullptr;
    }
}

/**
 * @brief Construct a new Dynamic Array< T>:: Copy constructor
 * 
 * @tparam T 
 * @param other other DynamicArray
 */
template <typename T>
DynamicArray<T>::DynamicArray(const DynamicArray<T> &other) : m_length(other.m_length), m_capacity(other.m_capacity), m_data(m_length > 0 ? new T[other.m_length] : nullptr)
{
    for (std::size_t i = 0; i  != m_length; i++)
    {
        m_data[i] = other.m_data[i];
    }
}

/**
 * @brief Copy assignment operator 
 * 
 * @tparam T 
 * @param other DynamicArray right hand side of the operator
 * @return  DynamicArray&
 */
template <typename T>
DynamicArray<T>& DynamicArray<T>::operator=(const DynamicArray<T> &other)
{
    if(this != &other)
    {
        if(m_length != other.m_length)
        {
            delete[] m_data;
            m_length = other.m_length;
            m_capacity = other.m_capacity;

            if(other.m_length > 0)
            {
                m_data = new T[other.m_length];
            }else
            {
                m_data = nullptr;
            }
        }
        for(std::size_t i = 0; i != m_length; i++)
        {
            m_data[i] = other.m_data[i];
        }
    }
    return *this;
}

/**
 * @brief Move constructor.
 * 
 * @tparam T 
 * @param other 
 */
template <typename T>
DynamicArray<T>::DynamicArray(DynamicArray &&other) noexcept :
    m_length(std::move(other.m_length)),
    m_capacity(std::move(other.m_capacity)),
    m_data(std::move(other.m_data))
{
    other.m_length = 0;
    other.m_capacity = 0;
    other.m_data = nullptr;

    std::cout << "Move constructor - other.m_length = " 
            << other.m_length 
            << " - this.m_length = "  
            << m_length 
            << std::endl;
}

/**
 * @brief Move assignment operator.
 * 
 * @tparam T 
 * @param other 
 * @return DynamicArray<T>& 
 */
template <typename T>
DynamicArray<T>& DynamicArray<T>::operator=(DynamicArray &&other) noexcept
{
    if(this != &other)
    {
        delete[] m_data;

        m_length = std::move(other.m_length);
        m_capacity = std::move(other.m_capacity);
        m_data = std::move(other.m_data);

        other.m_length = 0;
        other.m_capacity = 0;
        other.m_data = nullptr;
    }

    std::cout << "Move assignment operator - other.m_length = " 
            << other.m_length 
            << " - this.m_length = "  
            << m_length 
            << std::endl;

    return *this;
}

/**
 * @brief Push backs the *value* at the end of the array.
 *
 * @param dynmaic_array The dynamic array.
 * @param value The value to append to the array.
 */
template <typename T>
void DynamicArray<T>::push_back(const T &value)
{
    if (m_length == m_capacity)
    {
        m_capacity *= 2;

        T *temp = nullptr;
        temp = new T[m_capacity];

        for (std::size_t i = 0; i < m_length; i++)
        {
            temp[i] = m_data[i];
        }

        delete[] m_data;
        m_data = temp;
    }

    m_data[m_length] = value;
    m_length++;
}

/**
 * @brief Pop backs the value at the end of the vector.
 */
template <typename T>
void DynamicArray<T>::pop_back()
{
    if (m_length > 0)
    {
        m_length--;

        if (m_length < (m_capacity / 2))
        {
            m_capacity /= 2;

            T *temp = nullptr;
            temp = new T[m_capacity];

            for (std::size_t i = 0; i < m_length; i++)
            {
                temp[i] = m_data[i];
            }

            delete[] m_data;
            m_data = temp;
        }
    }
}

template <typename T>
T &DynamicArray<T>::operator[](const std::size_t index)
{
    return m_data[index];
}

template <typename T>
const T &DynamicArray<T>::operator[](const std::size_t index) const
{
    return m_data[index];
}

template <typename T>
std::size_t DynamicArray<T>::get_length() const
{
    return m_length;
}