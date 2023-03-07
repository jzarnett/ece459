Answers for the activity

If only look at the tables, we can consider performing *selection early* and
*projection early*.

If we look at the indexes, we may notice there is an `(customer\_id, city,
customer\_name)` index which contains the necessary information we need. That
means we can scan the index instead of scanning the table and dropping the
`customer\_email`.

If we look at the statistics, we may notice there are only few purchases with
over 5000 dollars whereas there are more customers who live in New York. It may
suggest that we may want to scan `orders` first, then join `customer` (or
index). Of course, only if the statistics are up-to-date.

The overhead is that the query optimization is applied in runtime. If the
original query is efficient enough, then the process may slow it down.

