double sum(double *array, int length)
{
    double total = 0;
    for (int i = 0; i < length; i++)
        total += array[i];
    return total;
}
