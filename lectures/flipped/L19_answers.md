Answers for the "how do you optimize it?" activity:

If only looking at the tables, we can consider performing *selection early* and
*projection early*.

* If we look at the index, we may notice there is a `(customer_id, city,
customer_name)` index containing the necessary information we need. That means
we can scan the index instead of scanning every full record from the `customer`
table and then drop the `customer_email`.

* If we look at the statistics, we may notice there are only few purchases with
over 5000 dollars whereas there are more customers who live in New York. It may
suggest that we may want to scan `orders` first, then join `customer` or its
index (assuming tables have similar number of records). Of course, only if the
statistics are up-to-date.

* The overhead is that the query optimization is applied at runtime. If the
original query is fast, then the optimization process may slow it down. Longer
discussions can be about situations with insufficient memory, caching enabled,
etc. Some optimizations may not work as expected.
