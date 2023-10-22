# obAnalytics

[![License](http://img.shields.io/badge/license-GPL%20%28%3E=%202%29-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-2.0.html)

Limit Order Book event processing and visualisation.

!["limit order book analytics"](https://raw.githubusercontent.com/phil8192/ob-analytics/master/ob-analytics.png "limit order book analytics") 

__obAnalytics__ is an Python package (translated from the original R package) intended for visualisation and analysis of limit
order data. The package is experimental and is based on the R code used to
create the visualizations in this original [Limit Order Book Visualisation](http://parasec.net/transmission/order-book-visualisation/) 
article.

## Installation
```
```

### Github
```
```

## Example use
Preprocessed limit order data from the inst/extdata directory has been included
in the package. The data, taken from a Bitcoin exchange on 2015-05-01, consists 
of 50,393 limit order events and 482 trades occurring from midnight up until 
~5am. 

The lob.data data structure contains 4 pandas data frames describing limit order 
events, trades, depth and summary statistics. All of which are described in 
detail in the package documentation. To visualize all of the example order book
data, use the plotPriceLevels function:

```
```

## Documentation


### Example use of obAnalytics package (html) 


### Example use (pdf)


### Manual 


## License

GPL (>= 2)

