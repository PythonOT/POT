/* -*- mode: C++; indent-tabs-mode: nil; -*-
 *
 *
 * This file has been adapted by Nicolas Bonneel (2013),
 * from network_simplex.h from LEMON, a generic C++ optimization library,
 * to implement a lightweight network simplex for mass transport, more
 * memory efficient that the original file. A previous version of this file
 * is used as part of the Displacement Interpolation project,
 * Web: http://www.cs.ubc.ca/labs/imager/tr/2011/DisplacementInterpolation/
 *
 *
 **** Original file Copyright Notice :
 *
 * Copyright (C) 2003-2010
 * Egervary Jeno Kombinatorikus Optimalizalasi Kutatocsoport
 * (Egervary Research Group on Combinatorial Optimization, EGRES).
 *
 * Permission to use, modify and distribute this software is granted
 * provided that this copyright notice appears in all copies. For
 * precise terms see the accompanying LICENSE file.
 *
 * This software is provided "AS IS" with no warranty of any kind,
 * express or implied, and with no claim as to its suitability for any
 * purpose.
 *
 */

#pragma once
#undef DEBUG_LVL
#undef EPSILON
#undef _EPSILON
#define EPSILON 2.2204460492503131e-15
#define _EPSILON 1e-8


/// \ingroup min_cost_flow_algs
///
/// \file
/// \brief Network Simplex algorithm for finding a minimum cost flow.

// if your compiler has troubles with stdext or hashmaps, just comment the following line to use a slower std::map instead
//#define HASHMAP

#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <queue>
#include <stack>
#ifdef HASHMAP
#include <hash_map>
#else
#include <map>
#endif
#include <cmath>
#include <cstring>
//#include "core.h"
//#include "lmath.h"

//#include "sparse_array_n.h"
#include "full_bipartitegraph.h"

#undef INVALIDNODE
#undef INVALID
#define INVALIDNODE -1
#define INVALID (-1)

namespace lemon {


    template <typename T>
	class ProxyObject;

	template<typename T>
	class SparseValueVector
	{
	public:
		SparseValueVector(size_t n=0)
		{
		}
		void resize(size_t n=0){};
        T operator[](const size_t id) const
		{
#ifdef HASHMAP
            typename stdext::hash_map<size_t,T>::const_iterator it = data.find(id);
#else
            typename std::map<size_t,T>::const_iterator it = data.find(id);
#endif
			if (it==data.end())
				return 0;
			else
				return it->second;
		}

		ProxyObject<T> operator[](const size_t id)
		{
			return ProxyObject<T>( this, id );
		}

        //private:
#ifdef HASHMAP
        stdext::hash_map<size_t,T> data;
#else
        std::map<size_t,T> data;
#endif

	};

	template <typename T>
	class ProxyObject {
	public:
        ProxyObject( SparseValueVector<T> *v, size_t idx ){_v=v; _idx=idx;};
		ProxyObject<T> & operator=( const T &v ) {
			// If we get here, we know that operator[] was called to perform a write access,
			// so we can insert an item in the vector if needed
			if (v!=0)
				_v->data[_idx]=v;
			return *this;
		}

		operator T() {
			// If we get here, we know that operator[] was called to perform a read access,
			// so we can simply return the existing object
#ifdef HASHMAP
            typename stdext::hash_map<size_t,T>::iterator it = _v->data.find(_idx);
#else
            typename std::map<size_t,T>::iterator it = _v->data.find(_idx);
#endif
			if (it==_v->data.end())
				return 0;
			else
				return it->second;
		}

		void operator+=(T val)
		{
			if (val==0) return;
#ifdef HASHMAP
            typename stdext::hash_map<size_t,T>::iterator it = _v->data.find(_idx);
#else
            typename std::map<size_t,T>::iterator it = _v->data.find(_idx);
#endif
			if (it==_v->data.end())
				_v->data[_idx] = val;
			else
			{
				T sum = it->second + val;
				if (sum==0)
					_v->data.erase(it);
				else
					it->second = sum;
			}
		}
		void operator-=(T val)
		{
			if (val==0) return;
#ifdef HASHMAP
            typename stdext::hash_map<size_t,T>::iterator it = _v->data.find(_idx);
#else
            typename std::map<size_t,T>::iterator it = _v->data.find(_idx);
#endif
			if (it==_v->data.end())
				_v->data[_idx] = -val;
			else
			{
				T sum = it->second - val;
				if (sum==0)
					_v->data.erase(it);
				else
					it->second = sum;
			}
		}

		SparseValueVector<T> *_v;
        size_t _idx;
	};



    /// \addtogroup min_cost_flow_algs
    /// @{

    /// \brief Implementation of the primal Network Simplex algorithm
    /// for finding a \ref min_cost_flow "minimum cost flow".
    ///
    /// \ref NetworkSimplexSimple implements the primal Network Simplex algorithm
    /// for finding a \ref min_cost_flow "minimum cost flow"
    /// \ref amo93networkflows, \ref dantzig63linearprog,
    /// \ref kellyoneill91netsimplex.
    /// This algorithm is a highly efficient specialized version of the
    /// linear programming simplex method directly for the minimum cost
    /// flow problem.
    ///
    /// In general, %NetworkSimplexSimple is the fastest implementation available
    /// in LEMON for this problem.
    /// Moreover, it supports both directions of the supply/demand inequality
    /// constraints. For more information, see \ref SupplyType.
    ///
    /// Most of the parameters of the problem (except for the digraph)
    /// can be given using separate functions, and the algorithm can be
    /// executed using the \ref run() function. If some parameters are not
    /// specified, then default values will be used.
    ///
    /// \tparam GR The digraph type the algorithm runs on.
    /// \tparam V The number type used for flow amounts, capacity bounds
    /// and supply values in the algorithm. By default, it is \c int64_t.
    /// \tparam C The number type used for costs and potentials in the
    /// algorithm. By default, it is the same as \c V.
    ///
    /// \warning Both number types must be signed and all input data must
    /// be integer.
    ///
    /// \note %NetworkSimplexSimple provides five different pivot rule
    /// implementations, from which the most efficient one is used
    /// by default. For more information, see \ref PivotRule.
    template <typename GR, typename V = int, typename C = V, typename NodesType = unsigned short int, typename ArcsType = int64_t>
    class NetworkSimplexSimple
    {
    public:

        /// \brief Constructor.
        ///
        /// The constructor of the class.
        ///
        /// \param graph The digraph the algorithm runs on.
        /// \param arc_mixing Indicate if the arcs have to be stored in a
        /// mixed order in the internal data structure.
        /// In special cases, it could lead to better overall performance,
        /// but it is usually slower. Therefore it is disabled by default.
        NetworkSimplexSimple(const GR& graph, bool arc_mixing, int nbnodes, ArcsType nb_arcs, uint64_t maxiters) :
        _graph(graph),  //_arc_id(graph),
        _arc_mixing(arc_mixing), _init_nb_nodes(nbnodes), _init_nb_arcs(nb_arcs),
        MAX(std::numeric_limits<Value>::max()),
        INF(std::numeric_limits<Value>::has_infinity ?
            std::numeric_limits<Value>::infinity() : MAX),
        _lazy_cost(false), _coords_a(nullptr), _coords_b(nullptr), _dim(0), _metric(0), _n1(0), _n2(0),
        _dense_cost(false), _D_ptr(nullptr), _D_n2(0),
        _warmstart_provided(false), _warmstart_tree_built(false)
        {
            // Reset data structures
            reset();
            max_iter = maxiters;
        }

        /// The type of the flow amounts, capacity bounds and supply values
        typedef V Value;
        /// The type of the arc costs
        typedef C Cost;

    public:

        /// \brief Problem type constants for the \c run() function.
        ///
        /// Enum type containing the problem type constants that can be
        /// returned by the \ref run() function of the algorithm.
        enum ProblemType {
            /// The problem has no feasible solution (flow).
            INFEASIBLE,
            /// The problem has optimal solution (i.e. it is feasible and
            /// bounded), and the algorithm has found optimal flow and node
            /// potentials (primal and dual solutions).
            OPTIMAL,
            /// The objective function of the problem is unbounded, i.e.
            /// there is a directed cycle having negative total cost and
            /// infinite upper bound.
            UNBOUNDED,
			/// The maximum number of iteration has been reached
			MAX_ITER_REACHED
        };

        /// \brief Constants for selecting the type of the supply constraints.
        ///
        /// Enum type containing constants for selecting the supply type,
        /// i.e. the direction of the inequalities in the supply/demand
        /// constraints of the \ref min_cost_flow "minimum cost flow problem".
        ///
        /// The default supply type is \c GEQ, the \c LEQ type can be
        /// selected using \ref supplyType().
        /// The equality form is a special case of both supply types.
        enum SupplyType {
            /// This option means that there are <em>"greater or equal"</em>
            /// supply/demand constraints in the definition of the problem.
            GEQ,
            /// This option means that there are <em>"less or equal"</em>
            /// supply/demand constraints in the definition of the problem.
            LEQ
        };



    private:

        uint64_t max_iter;
        TEMPLATE_DIGRAPH_TYPEDEFS(GR);

        typedef std::vector<int> IntVector;
        typedef std::vector<ArcsType> ArcVector;
        typedef std::vector<Value> ValueVector;
        typedef std::vector<Cost> CostVector;
        //	typedef SparseValueVector<Cost> CostVector;
        typedef std::vector<char> BoolVector;
        // Note: vector<char> is used instead of vector<bool> for efficiency reasons

        // State constants for arcs
        enum ArcState {
            STATE_UPPER = -1,
            STATE_TREE  =  0,
            STATE_LOWER =  1
        };

        typedef std::vector<signed char> StateVector;
        // Note: vector<signed char> is used instead of vector<ArcState> for
        // efficiency reasons

    private:

        // Data related to the underlying digraph
        const GR &_graph;
        int _node_num;
        int _n1;  // Number of source nodes (for lazy cost computation)
        int _n2;  // Number of target nodes (for lazy cost computation)
        ArcsType _arc_num;
        ArcsType _all_arc_num;
        ArcsType _search_arc_num;

        // Parameters of the problem
        SupplyType _stype;
        Value _sum_supply;

        inline int _node_id(int n) const {return _node_num-n-1;} ;

// 	    IntArcMap _arc_id;
        IntVector _source;  // keep nodes as integers
        IntVector _target;
        bool _arc_mixing;
    public:
        // Node and arc data
        CostVector _cost;
        ValueVector _supply;
        ValueVector _flow;
        //SparseValueVector<Value> _flow;
        CostVector _pi;

        // Lazy cost computation support
        bool _lazy_cost;
        const double* _coords_a;
        const double* _coords_b;
        int _dim;
        int _metric; // 0: sqeuclidean, 1: euclidean, 2: cityblock

        // Dense cost matrix pointer (lazy access, no copy)
        bool _dense_cost;
        const double* _D_ptr;  // pointer to row-major cost matrix
        int _D_n2;             // number of columns in D (original n2)

    private:
        // Warmstart data
        bool _warmstart_provided;  // Flag indicating warmstart is available
        bool _warmstart_tree_built;  // Flag: tree was built by warmstartInit()

        // Data for storing the spanning tree structure
        IntVector _parent;
        ArcVector _pred;
        IntVector _thread;
        IntVector _rev_thread;
        IntVector _succ_num;
        IntVector _last_succ;
        IntVector _dirty_revs;
        BoolVector _forward;
        StateVector _state;
        ArcsType _root;

        // Temporary data used in the current pivot iteration
        ArcsType in_arc, join, u_in, v_in, u_out, v_out;
        ArcsType first, second, right, last;
        ArcsType stem, par_stem, new_stem;
        Value delta;

        const Value MAX;

        ArcsType mixingCoeff;

    public:

        /// \brief Constant for infinite upper bounds (capacities).
        ///
        /// Constant for infinite upper bounds (capacities).
        /// It is \c std::numeric_limits<Value>::infinity() if available,
        /// \c std::numeric_limits<Value>::max() otherwise.
        const Value INF;

    private:

        // thank you to DVK and MizardX from StackOverflow for this function!
        inline ArcsType sequence(ArcsType k) const {
            ArcsType smallv = (k > num_total_big_subsequence_numbers) & 1;

            k -= num_total_big_subsequence_numbers * smallv;
            ArcsType subsequence_length2 = subsequence_length- smallv;
            ArcsType subsequence_num = (k / subsequence_length2) + num_big_subseqiences * smallv;
            ArcsType subsequence_offset = (k % subsequence_length2) * mixingCoeff;

            return subsequence_offset + subsequence_num;
        }
        ArcsType subsequence_length;
        ArcsType num_big_subseqiences;
        ArcsType num_total_big_subsequence_numbers;

        inline ArcsType getArcID(const Arc &arc) const
        {
            //int n = _arc_num-arc._id-1;
            ArcsType n = _arc_num-GR::id(arc)-1;

            //ArcsType a = mixingCoeff*(n%mixingCoeff) + n/mixingCoeff;
            //ArcsType b = _arc_id[arc];
            if (_arc_mixing)
                return sequence(n);
            else
                return n;
        }

        // finally unused because too slow
        inline ArcsType getSource(const ArcsType arc) const
        {
            //ArcsType a = _source[arc];
            //return a;

            ArcsType n = _arc_num-arc-1;
            if (_arc_mixing)
                n = mixingCoeff*(n%mixingCoeff) + n/mixingCoeff;

            ArcsType b;
            if (n>=0)
                b = _node_id(_graph.source(GR::arcFromId( n ) ));
            else
            {
                n = arc+1-_arc_num;
                if ( n<=_node_num)
                    b = _node_num;
                else
                    if ( n>=_graph._n1)
                        b = _graph._n1;
                    else
                        b = _graph._n1-n;
            }

            return b;
        }



        // Implementation of the Block Search pivot rule
        class BlockSearchPivotRule
        {
        private:

            // References to the NetworkSimplexSimple class
            const IntVector  &_source;
            const IntVector  &_target;
            const CostVector &_cost;
            const StateVector &_state;
            const CostVector &_pi;
            ArcsType &_in_arc;
            ArcsType _search_arc_num;

            // Pivot rule data
            ArcsType _block_size;
            ArcsType _next_arc;
            NetworkSimplexSimple &_ns;

        public:

            // Constructor
            BlockSearchPivotRule(NetworkSimplexSimple &ns) :
            _source(ns._source), _target(ns._target),
            _cost(ns._cost), _state(ns._state), _pi(ns._pi),
            _in_arc(ns.in_arc), _search_arc_num(ns._search_arc_num),
            _next_arc(0),_ns(ns)
            {
                // The main parameters of the pivot rule
                const double BLOCK_SIZE_FACTOR = 1.0;
                const ArcsType MIN_BLOCK_SIZE = 10;

                _block_size = std::max(ArcsType(BLOCK_SIZE_FACTOR * std::sqrt(double(_search_arc_num))), MIN_BLOCK_SIZE);
            }

            // Get cost for an arc (either from pre-computed array or compute lazily)
            inline Cost getCost(ArcsType e) const {
                if (_ns._dense_cost) {
                    // Dense matrix mode: read directly from D pointer
                    return _ns._D_ptr[_ns._arc_num - e - 1];
                } else if (!_ns._lazy_cost) {
                    return _cost[e];
                } else {
                    // For lazy mode, compute cost from coordinates inline
                    // _source and _target use reversed node numbering
                    int i = _ns._node_num - _source[e] - 1;
                    int j = _ns._node_num - _target[e] - 1 - _ns._n1;
                    
                    const double* xa = _ns._coords_a + i * _ns._dim;
                    const double* xb = _ns._coords_b + j * _ns._dim;
                    Cost cost = 0;
                    
                    if (_ns._metric == 0) {  // sqeuclidean
                        for (int d = 0; d < _ns._dim; ++d) {
                            Cost diff = xa[d] - xb[d];
                            cost += diff * diff;
                        }
                        return cost;
                    } else if (_ns._metric == 1) {  // euclidean
                        for (int d = 0; d < _ns._dim; ++d) {
                            Cost diff = xa[d] - xb[d];
                            cost += diff * diff;
                        }
                        return std::sqrt(cost);
                    } else {  // cityblock
                        for (int d = 0; d < _ns._dim; ++d) {
                            cost += std::abs(xa[d] - xb[d]);
                        }
                        return cost;
                    }
                }
            }

            // Find next entering arc
            bool findEnteringArc() {
                Cost c, min = 0;
                ArcsType e;
                ArcsType cnt = _block_size;
                double a;
                    for (e = _next_arc; e != _search_arc_num; ++e) {
                        c = _state[e] * (getCost(e) + _pi[_source[e]] - _pi[_target[e]]);
                        if (c < min) {
                            min = c;
                            _in_arc = e;
                        }
                        if (--cnt == 0) {
                            a=fabs(_pi[_source[_in_arc]])>fabs(_pi[_target[_in_arc]]) ? fabs(_pi[_source[_in_arc]]):fabs(_pi[_target[_in_arc]]);
                            a=a>fabs(getCost(_in_arc))?a:fabs(getCost(_in_arc));
                            if (min <  -EPSILON*a) goto search_end;
                            cnt = _block_size;
                        }
                    }
                    for (e = 0; e != _next_arc; ++e) {
                        c = _state[e] * (getCost(e) + _pi[_source[e]] - _pi[_target[e]]);
                        if (c < min) {
                            min = c;
                            _in_arc = e;
                        }
                        if (--cnt == 0) {
                            a=fabs(_pi[_source[_in_arc]])>fabs(_pi[_target[_in_arc]]) ? fabs(_pi[_source[_in_arc]]):fabs(_pi[_target[_in_arc]]);
                            a=a>fabs(getCost(_in_arc))?a:fabs(getCost(_in_arc));
                            if (min <  -EPSILON*a) goto search_end;
                            cnt = _block_size;
                        }
                    }
                    a=fabs(_pi[_source[_in_arc]])>fabs(_pi[_target[_in_arc]]) ? fabs(_pi[_source[_in_arc]]):fabs(_pi[_target[_in_arc]]);
                    a=a>fabs(getCost(_in_arc))?a:fabs(getCost(_in_arc));
                    if (min >=  -EPSILON*a) return false;

            search_end:
                _next_arc = e;
                return true;
            }

        }; //class BlockSearchPivotRule



    public:

        // Public accessors for efficient result extraction
        ArcsType arcNum() const { return _arc_num; }
        int nodeNum() const { return _node_num; }
        int n1() const { return _n1; }
        int n2() const { return _n2; }
        Cost pi(int internal_node) const { return _pi[internal_node]; }

        int _init_nb_nodes;
        ArcsType _init_nb_arcs;

        /// \name Parameters
        /// The parameters of the algorithm can be specified using these
        /// functions.

        /// @{


        /// \brief Set the costs of the arcs.
        ///
        /// This function sets the costs of the arcs.
        /// If it is not used before calling \ref run(), the costs
        /// will be set to \c 1 on all arcs.
        ///
        /// \param map An arc map storing the costs.
        /// Its \c Value type must be convertible to the \c Cost type
        /// of the algorithm.
        ///
        /// \return <tt>(*this)</tt>
        template<typename CostMap>
        NetworkSimplexSimple& costMap(const CostMap& map) {
            Arc a; _graph.first(a);
            for (; a != INVALID; _graph.next(a)) {
                _cost[getArcID(a)] = map[a];
            }
            return *this;
        }


        /// \brief Set the costs of one arc.
        ///
        /// This function sets the costs of one arcs.
        /// Done for memory reasons
        ///
        /// \param arc An arc.
        /// \param arc A cost
        ///
        /// \return <tt>(*this)</tt>
        template<typename Value>
        NetworkSimplexSimple& setCost(const Arc& arc, const Value cost) {
            _cost[getArcID(arc)] = cost;
            return *this;
        }

        /// \brief Enable lazy cost computation from coordinates.
        ///
        /// This function enables lazy cost computation where distances are
        /// computed on-the-fly from point coordinates instead of using a
        /// pre-computed cost matrix.
        ///
        /// \param coords_a Pointer to source coordinates (n1 x dim array)
        /// \param coords_b Pointer to target coordinates (n2 x dim array)
        /// \param dim Dimension of the coordinates
        /// \param metric Distance metric: 0=sqeuclidean, 1=euclidean, 2=cityblock
        ///
        /// \return <tt>(*this)</tt>
        NetworkSimplexSimple& setLazyCost(const double* coords_a, const double* coords_b, 
                                           int dim, int metric, int n1, int n2) {
            _lazy_cost = true;
            _coords_a = coords_a;
            _coords_b = coords_b;
            _dim = dim;
            _metric = metric;
            _n1 = n1;
            _n2 = n2;
            return *this;
        }

        /// \brief Set a dense cost matrix pointer for lazy access.
        ///
        /// This function stores a pointer to the cost matrix D (row-major)
        /// so that costs can be read directly without copying.
        /// Requires arc_mixing=false and n==n1, m==n2 (no zero-mass filtering).
        ///
        /// \param D Pointer to the n1 x n2 cost matrix (row-major)
        /// \param n2 Number of columns in D
        ///
        /// \return <tt>(*this)</tt>
        NetworkSimplexSimple& setDenseCostMatrix(const double* D, int n2) {
            _dense_cost = true;
            _D_ptr = D;
            _D_n2 = n2;
            return *this;
        }

        /// \brief Compute cost lazily from coordinates.
        ///
        /// Computes the distance between source node i and target node j
        /// based on the specified metric.
        ///
        /// \param i Source node index
        /// \param j Target node index (adjusted by subtracting n1)
        ///
        /// \return Cost (distance) between the two points
        inline Cost computeLazyCost(int i, int j_adjusted) const {
            const double* xa = _coords_a + i * _dim;
            const double* xb = _coords_b + j_adjusted * _dim;
            Cost cost = 0;
            
            if (_metric == 0) {  // sqeuclidean
                for (int d = 0; d < _dim; ++d) {
                    Cost diff = xa[d] - xb[d];
                    cost += diff * diff;
                }
                return cost;
            } else if (_metric == 1) {  // euclidean
                for (int d = 0; d < _dim; ++d) {
                    Cost diff = xa[d] - xb[d];
                    cost += diff * diff;
                }
                return std::sqrt(cost);
            } else {  // cityblock (L1)
                for (int d = 0; d < _dim; ++d) {
                    cost += std::abs(xa[d] - xb[d]);
                }
                return cost;
            }
        }


        /// \brief Get cost for an arc (either from array or compute lazily).
        ///
        /// This is the main cost accessor that works from anywhere in the class.
        /// In lazy mode, computes cost on-the-fly from coordinates.
        /// In normal mode, returns pre-computed cost from array.
        ///
        /// \param arc_id The arc ID
        /// \return Cost of the arc
        inline Cost getCostForArc(ArcsType arc_id) const {
            if (_dense_cost) {
                // Dense matrix mode: read directly from D pointer
                // For artificial arcs (>= _arc_num), read from _cost array
                if (arc_id >= _arc_num) {
                    return _cost[arc_id];
                }
                // Without arc mixing: internal arc_id maps to graph arc = _arc_num - arc_id - 1
                // graph arc g encodes source i = g / m, target j = g % m
                // cost = D[i * _D_n2 + j] = D[g] (since m == _D_n2)
                return _D_ptr[_arc_num - arc_id - 1];
            } else if (!_lazy_cost) {
                return _cost[arc_id];
            } else {
                // For artificial arcs (>= _arc_num), return 0
                if (arc_id >= _arc_num) {
                    return 0;
                }
                // Compute lazily from coordinates
                // Convert internal node IDs back to graph node IDs, then to coordinate indices
                int i = _node_num - _source[arc_id] - 1;  // graph source in [0, _n1-1]
                int j = _node_num - _target[arc_id] - 1 - _n1;  // graph target in [_n1, _node_num-1] -> [0, _n2-1]
                return computeLazyCost(i, j);
            }
        }

        /// \brief Set the supply values of the nodes.
        ///
        /// This function sets the supply values of the nodes.
        /// If neither this function nor \ref stSupply() is used before
        /// calling \ref run(), the supply of each node will be set to zero.
        ///
        /// \param map A node map storing the supply values.
        /// Its \c Value type must be convertible to the \c Value type
        /// of the algorithm.
        ///
        /// \return <tt>(*this)</tt>
        template<typename SupplyMap>
        NetworkSimplexSimple& supplyMap(const SupplyMap& map) {
            Node n; _graph.first(n);
            for (; n != INVALIDNODE; _graph.next(n)) {
                _supply[_node_id(n)] = map[n];
            }
            return *this;
        }
        template<typename SupplyMap>
        NetworkSimplexSimple& supplyMap(const SupplyMap* map1, int n1, const SupplyMap* map2, int n2) {
            Node n; _graph.first(n);
            for (; n != INVALIDNODE; _graph.next(n)) {
                if (n<n1)
                    _supply[_node_id(n)] = map1[n];
                else
                    _supply[_node_id(n)] = map2[n-n1];
            }
            return *this;
        }
        template<typename SupplyMap>
        NetworkSimplexSimple& supplyMapAll(SupplyMap val1, int n1, SupplyMap val2, int n2) {
            Node n; _graph.first(n);
            for (; n != INVALIDNODE; _graph.next(n)) {
                if (n<n1)
                    _supply[_node_id(n)] = val1;
                else
                    _supply[_node_id(n)] = val2;
            }
            return *this;
        }

        /// \brief Set single source and target nodes and a supply value.
        ///
        /// This function sets a single source node and a single target node
        /// and the required flow value.
        /// If neither this function nor \ref supplyMap() is used before
        /// calling \ref run(), the supply of each node will be set to zero.
        ///
        /// Using this function has the same effect as using \ref supplyMap()
        /// with such a map in which \c k is assigned to \c s, \c -k is
        /// assigned to \c t and all other nodes have zero supply value.
        ///
        /// \param s The source node.
        /// \param t The target node.
        /// \param k The required amount of flow from node \c s to node \c t
        /// (i.e. the supply of \c s and the demand of \c t).
        ///
        /// \return <tt>(*this)</tt>
        NetworkSimplexSimple& stSupply(const Node& s, const Node& t, Value k) {
            for (int i = 0; i != _node_num; ++i) {
                _supply[i] = 0;
            }
            _supply[_node_id(s)] =  k;
            _supply[_node_id(t)] = -k;
            return *this;
        }

        /// \brief Set the type of the supply constraints.
        ///
        /// This function sets the type of the supply/demand constraints.
        /// If it is not used before calling \ref run(), the \ref GEQ supply
        /// type will be used.
        ///
        /// For more information, see \ref SupplyType.
        ///
        /// \return <tt>(*this)</tt>
        NetworkSimplexSimple& supplyType(SupplyType supply_type) {
            _stype = supply_type;
            return *this;
        }

        /// \brief Set initial dual potentials for warmstart.
        ///
        /// This function sets warmstart dual potentials that will be used
        /// to guide the initial pivots in the network simplex algorithm.
        /// The potentials should come from a previous solution (e.g., Sinkhorn or EMD).
        ///
        /// \param alpha Source node potentials (size n), where alpha[i] = -pi[source_i]
        /// \param beta Target node potentials (size m), where beta[j] = +pi[target_j]
        /// \param n Number of source nodes (compressed, non-zero supply)
        /// \param m Number of target nodes (compressed, non-zero supply)
        ///
        void setWarmstartPotentials(const Cost* alpha, const Cost* beta, int n, int m) {
            // Graph source nodes: 0..n-1, stored at internal index _node_id(i)
            // Graph target nodes: n..n+m-1, stored at internal index _node_id(n+j)
            // _node_id(k) = _node_num - k - 1 (reversal mapping)
            // Note: warmstartInit() will refine these by recomputing from the tree structure.

            for (int i = 0; i < n; ++i) {
                _pi[_node_id(i)] = -alpha[i];  // pi[source] = -alpha
            }
            for (int j = 0; j < m; ++j) {
                _pi[_node_id(n + j)] = beta[j];  // pi[target] = +beta
            }
            _warmstart_provided = true;
        }

        /// @}

        /// \name Execution Control
        /// The algorithm can be executed using \ref run().

        /// @{

        /// \brief Run the algorithm.
        ///
        /// This function runs the algorithm.
        /// The paramters can be specified using functions \ref lowerMap(),
        /// \ref upperMap(), \ref costMap(), \ref supplyMap(), \ref stSupply(),
        /// \ref supplyType().
        /// For example,
        /// \code
        ///   NetworkSimplexSimple<ListDigraph> ns(graph);
        ///   ns.lowerMap(lower).upperMap(upper).costMap(cost)
        ///     .supplyMap(sup).run();
        /// \endcode
        ///
        /// This function can be called more than once. All the given parameters
        /// are kept for the next call, unless \ref resetParams() or \ref reset()
        /// is used, thus only the modified parameters have to be set again.
        /// If the underlying digraph was also modified after the construction
        /// of the class (or the last \ref reset() call), then the \ref reset()
        /// function must be called.
        ///
        /// \param pivot_rule The pivot rule that will be used during the
        /// algorithm. For more information, see \ref PivotRule.
        ///
        /// \return \c INFEASIBLE if no feasible flow exists,
        /// \n \c OPTIMAL if the problem has optimal solution
        /// (i.e. it is feasible and bounded), and the algorithm has found
        /// optimal flow and node potentials (primal and dual solutions),
        /// \n \c UNBOUNDED if the objective function of the problem is
        /// unbounded, i.e. there is a directed cycle having negative total
        /// cost and infinite upper bound.
        ///
        /// \see ProblemType, PivotRule
        /// \see resetParams(), reset()
        ProblemType run() {

            if (_warmstart_provided) {
                if (!warmstartInit()) return INFEASIBLE;
                _warmstart_tree_built = true;
            } else {
                if (!init()) return INFEASIBLE;
                _warmstart_tree_built = false;
            }

            return start();
        }

        /// \brief Reset all the parameters that have been given before.
        ///
        /// This function resets all the paramaters that have been given
        /// before using functions \ref lowerMap(), \ref upperMap(),
        /// \ref costMap(), \ref supplyMap(), \ref stSupply(), \ref supplyType().
        ///
        /// It is useful for multiple \ref run() calls. Basically, all the given
        /// parameters are kept for the next \ref run() call, unless
        /// \ref resetParams() or \ref reset() is used.
        /// If the underlying digraph was also modified after the construction
        /// of the class or the last \ref reset() call, then the \ref reset()
        /// function must be used, otherwise \ref resetParams() is sufficient.
        ///
        /// For example,
        /// \code
        ///   NetworkSimplexSimple<ListDigraph> ns(graph);
        ///
        ///   // First run
        ///   ns.lowerMap(lower).upperMap(upper).costMap(cost)
        ///     .supplyMap(sup).run();
        ///
        ///   // Run again with modified cost map (resetParams() is not called,
        ///   // so only the cost map have to be set again)
        ///   cost[e] += 100;
        ///   ns.costMap(cost).run();
        ///
        ///   // Run again from scratch using resetParams()
        ///   // (the lower bounds will be set to zero on all arcs)
        ///   ns.resetParams();
        ///   ns.upperMap(capacity).costMap(cost)
        ///     .supplyMap(sup).run();
        /// \endcode
        ///
        /// \return <tt>(*this)</tt>
        ///
        /// \see reset(), run()
        NetworkSimplexSimple& resetParams() {
            for (int i = 0; i != _node_num; ++i) {
                _supply[i] = 0;
            }
            for (ArcsType i = 0; i != _arc_num; ++i) {
                _cost[i] = 1;
            }
            _stype = GEQ;
            _warmstart_provided = false;
            _warmstart_tree_built = false;  // Reset warmstart flag
            return *this;
        }



        int64_t divid (int64_t x, int64_t y)
        {
            return (x-x%y)/y;
        }

        /// \brief Reset the internal data structures and all the parameters
        /// that have been given before.
        ///
        /// This function resets the internal data structures and all the
        /// paramaters that have been given before using functions \ref lowerMap(),
        /// \ref upperMap(), \ref costMap(), \ref supplyMap(), \ref stSupply(),
        /// \ref supplyType().
        ///
        /// It is useful for multiple \ref run() calls. Basically, all the given
        /// parameters are kept for the next \ref run() call, unless
        /// \ref resetParams() or \ref reset() is used.
        /// If the underlying digraph was also modified after the construction
        /// of the class or the last \ref reset() call, then the \ref reset()
        /// function must be used, otherwise \ref resetParams() is sufficient.
        ///
        /// See \ref resetParams() for examples.
        ///
        /// \return <tt>(*this)</tt>
        ///
        /// \see resetParams(), run()
        NetworkSimplexSimple& reset() {
            // Resize vectors
            _node_num = _init_nb_nodes;
            _arc_num = _init_nb_arcs;
            int all_node_num = _node_num + 1;
            ArcsType max_arc_num = _arc_num + 2 * _node_num;

            _source.resize(max_arc_num);
            _target.resize(max_arc_num);

            _cost.resize(max_arc_num);
            _supply.resize(all_node_num);
            _flow.resize(max_arc_num);
            _pi.resize(all_node_num);

            _parent.resize(all_node_num);
            _pred.resize(all_node_num);
            _forward.resize(all_node_num);
            _thread.resize(all_node_num);
            _rev_thread.resize(all_node_num);
            _succ_num.resize(all_node_num);
            _last_succ.resize(all_node_num);
            _state.resize(max_arc_num);


            //_arc_mixing=false;
            if (_arc_mixing) {
                // Store the arcs in a mixed order
                const ArcsType k = std::max(ArcsType(std::sqrt(double(_arc_num))), ArcsType(10));
                mixingCoeff = k;
                subsequence_length = _arc_num / mixingCoeff + 1;
                num_big_subseqiences = _arc_num % mixingCoeff;
                num_total_big_subsequence_numbers = subsequence_length * num_big_subseqiences;

                ArcsType i = 0, j = 0;
                Arc a; _graph.first(a);
                for (; a != INVALID; _graph.next(a)) {
                    _source[i] = _node_id(_graph.source(a));
                    _target[i] = _node_id(_graph.target(a));
                    //_arc_id[a] = i;
                    if ((i += k) >= _arc_num) i = ++j;
                }
            } else {
                // Store the arcs in the original order
                ArcsType i = 0;
                Arc a; _graph.first(a);
                for (; a != INVALID; _graph.next(a), ++i) {
                    _source[i] = _node_id(_graph.source(a));
                    _target[i] = _node_id(_graph.target(a));
                    //_arc_id[a] = i;
                }
            }

            // Reset parameters
            resetParams();
            return *this;
        }

        /// @}

        /// \name Query Functions
        /// The results of the algorithm can be obtained using these
        /// functions.\n
        /// The \ref run() function must be called before using them.

        /// @{

        /// \brief Return the total cost of the found flow.
        ///
        /// This function returns the total cost of the found flow.
        /// Its complexity is O(e).
        ///
        /// \note The return type of the function can be specified as a
        /// template parameter. For example,
        /// \code
        ///   ns.totalCost<double>();
        /// \endcode
        /// It is useful if the total cost cannot be stored in the \c Cost
        /// type of the algorithm, which is the default return type of the
        /// function.
        ///
        /// \pre \ref run() must be called before using this function.
        /*template <typename Number>
         Number totalCost() const {
         Number c = 0;
         for (ArcIt a(_graph); a != INVALID; ++a) {
         int64_t i = getArcID(a);
         c += Number(_flow[i]) * Number(_cost[i]);
         }
         return c;
         }*/

        template <typename Number>
        Number totalCost() const {
            Number c = 0;

            /*#ifdef HASHMAP
             typename stdext::hash_map<int64_t, Value>::const_iterator it;
             #else
             typename std::map<int64_t, Value>::const_iterator it;
             #endif
             for (it = _flow.data.begin(); it!=_flow.data.end(); ++it)
             c += Number(it->second) * Number(_cost[it->first]);
             return c;*/

            if (_dense_cost) {
                // Dense matrix mode: compute cost from D pointer
                for (ArcsType i=0; i<_flow.size(); i++) {
                    if (_flow[i] != 0) {
                        c += _flow[i] * Number(getCostForArc(i));
                    }
                }
            } else if (!_lazy_cost) {
                for (ArcsType i=0; i<_flow.size(); i++)
                    c += _flow[i] * Number(_cost[i]);
            } else {
                // Compute costs lazily
                for (ArcsType i=0; i<_flow.size(); i++) {
                    if (_flow[i] != 0) {
                        int src = _node_num - _source[i] - 1;
                        int tgt = _node_num - _target[i] - 1 - _n1;
                        c += _flow[i] * Number(computeLazyCost(src, tgt));
                    }
                }
            }
            return c;

        }

#ifndef DOXYGEN
        Cost totalCost() const {
            return totalCost<Cost>();
        }
#endif

        /// \brief Return the flow on the given arc.
        ///
        /// This function returns the flow on the given arc.
        ///
        /// \pre \ref run() must be called before using this function.
        Value flow(const Arc& a) const {
            return _flow[getArcID(a)];
        }

        /// \brief Return the flow map (the primal solution).
        ///
        /// This function copies the flow value on each arc into the given
        /// map. The \c Value type of the algorithm must be convertible to
        /// the \c Value type of the map.
        ///
        /// \pre \ref run() must be called before using this function.
        template <typename FlowMap>
        void flowMap(FlowMap &map) const {
            Arc a; _graph.first(a);
            for (; a != INVALID; _graph.next(a)) {
                map.set(a, _flow[getArcID(a)]);
            }
        }

        /// \brief Return the potential (dual value) of the given node.
        ///
        /// This function returns the potential (dual value) of the
        /// given node.
        ///
        /// \pre \ref run() must be called before using this function.
        Cost potential(const Node& n) const {
            return _pi[_node_id(n)];
        }

        /// \brief Return the potential map (the dual solution).
        ///
        /// This function copies the potential (dual value) of each node
        /// into the given map.
        /// The \c Cost type of the algorithm must be convertible to the
        /// \c Value type of the map.
        ///
        /// \pre \ref run() must be called before using this function.
        template <typename PotentialMap>
        void potentialMap(PotentialMap &map) const {
            Node n; _graph.first(n);
            for (; n != INVALID; _graph.next(n)) {
                map.set(n, _pi[_node_id(n)]);
            }
        }

        /// @}

    private:

        // WARMSTART: Build spanning tree from dual potentials
        bool warmstartInit() {
            if (_node_num == 0) return false;

            // Check supply balance
            _sum_supply = 0;
            for (int i = 0; i != _node_num; ++i) {
                _sum_supply += _supply[i];
            }
            if (fabs(_sum_supply) > _EPSILON) return false;
            _sum_supply = 0;
            int tree_edges = 0;
            std::vector<ArcsType> tree_arcs;
            tree_arcs.reserve(_node_num);
            Cost ART_COST = 0;

            {
                ArcsType K = std::min((ArcsType)(4 * _node_num), _arc_num);

                // Max-heap: (|reduced_cost|, arc_index).  We keep the K smallest.

                typedef std::pair<Cost, ArcsType> HeapEntry;
                std::priority_queue<HeapEntry> maxheap;

                for (ArcsType e = 0; e < _arc_num; ++e) {
                    _state[e] = STATE_LOWER;
                    Cost c = getCostForArc(e);
                    if (c > ART_COST) ART_COST = c;
                    Cost rc = fabs(c + _pi[_source[e]] - _pi[_target[e]]);
                    if ((ArcsType)maxheap.size() < K) {
                        maxheap.push({rc, e});
                    } else if (rc < maxheap.top().first) {
                        maxheap.pop();
                        maxheap.push({rc, e});
                    }
                }
                if (std::numeric_limits<Cost>::is_exact) {
                    ART_COST = std::numeric_limits<Cost>::max() / 2 + 1;
                } else {
                    ART_COST = (ART_COST + 1) * _node_num;
                }

                std::vector<HeapEntry> candidates;
                candidates.reserve(maxheap.size());
                while (!maxheap.empty()) {
                    candidates.push_back(maxheap.top());
                    maxheap.pop();
                }
                
                std::sort(candidates.begin(), candidates.end(),
                    [](const HeapEntry& a, const HeapEntry& b) {
                        return a.first < b.first;
                    });

                // Kruskal's MST with union-find
                std::vector<int> uf_parent(_node_num);
                std::vector<int> uf_rank(_node_num, 0);
                for (int i = 0; i < _node_num; ++i) uf_parent[i] = i;

                for (ArcsType idx = 0; idx < (ArcsType)candidates.size() && tree_edges < _node_num - 1; ++idx) {
                    ArcsType e = candidates[idx].second;
                    int s = _source[e];
                    int t = _target[e];
                    int rs = s, rt = t;
                    while (uf_parent[rs] != rs) { uf_parent[rs] = uf_parent[uf_parent[rs]]; rs = uf_parent[rs]; }
                    while (uf_parent[rt] != rt) { uf_parent[rt] = uf_parent[uf_parent[rt]]; rt = uf_parent[rt]; }
                    if (rs == rt) continue;
                    if (uf_rank[rs] < uf_rank[rt]) std::swap(rs, rt);
                    uf_parent[rt] = rs;
                    if (uf_rank[rs] == uf_rank[rt]) uf_rank[rs]++;
                    tree_arcs.push_back(e);
                    tree_edges++;
                }

                // Fallback: if K best weren't enough to span, scan remaining arcs
                if (tree_edges < _node_num - 1) {
                    std::vector<bool> considered(_arc_num, false);
                    for (auto& c : candidates) considered[c.second] = true;

                    for (ArcsType e = 0; e < _arc_num && tree_edges < _node_num - 1; ++e) {
                        if (considered[e]) continue;
                        int s = _source[e];
                        int t = _target[e];
                        int rs = s, rt = t;
                        while (uf_parent[rs] != rs) { uf_parent[rs] = uf_parent[uf_parent[rs]]; rs = uf_parent[rs]; }
                        while (uf_parent[rt] != rt) { uf_parent[rt] = uf_parent[uf_parent[rt]]; rt = uf_parent[rt]; }
                        if (rs == rt) continue;
                        if (uf_rank[rs] < uf_rank[rt]) std::swap(rs, rt);
                        uf_parent[rt] = rs;
                        if (uf_rank[rs] == uf_rank[rt]) uf_rank[rs]++;
                        tree_arcs.push_back(e);
                        tree_edges++;
                    }
                }
            }

            std::vector<int> tree_adj_deg(_node_num, 0);
            for (int k = 0; k < tree_edges; ++k) {
                ArcsType e = tree_arcs[k];
                tree_adj_deg[_source[e]]++;
                tree_adj_deg[_target[e]]++;
            }
            std::vector<int> tree_adj_start(_node_num + 1, 0);
            for (int i = 0; i < _node_num; ++i) {
                tree_adj_start[i + 1] = tree_adj_start[i] + tree_adj_deg[i];
            }
            int total_adj = tree_adj_start[_node_num];
            std::vector<int> tree_adj_node(total_adj);
            std::vector<ArcsType> tree_adj_arc(total_adj);
            std::vector<int> tree_adj_pos(_node_num, 0);
            for (int k = 0; k < tree_edges; ++k) {
                ArcsType e = tree_arcs[k];
                int s = _source[e], t = _target[e];
                int ps = tree_adj_start[s] + tree_adj_pos[s]++;
                tree_adj_node[ps] = t;
                tree_adj_arc[ps] = e;
                int pt = tree_adj_start[t] + tree_adj_pos[t]++;
                tree_adj_node[pt] = s;
                tree_adj_arc[pt] = e;
            }

            // STEP 2: Set up artificial arcs
            _search_arc_num = _arc_num;
            _all_arc_num = _arc_num + _node_num;
            _root = _node_num;

            for (ArcsType u = 0, e = _arc_num; u != _node_num; ++u, ++e) {
                _state[e] = STATE_TREE;
                if (_supply[u] >= 0) {
                    _source[e] = u;
                    _target[e] = _root;
                    _cost[e] = 0;
                    _flow[e] = _supply[u];
                } else {
                    _source[e] = _root;
                    _target[e] = u;
                    _cost[e] = ART_COST;
                    _flow[e] = -_supply[u];
                }
            }

            // Root node setup
            _parent[_root] = -1;
            _pred[_root] = -1;
            _supply[_root] = -_sum_supply;
            _pi[_root] = 0;

            // STEP 3: BFS from root to build tree structure
            std::vector<bool> is_rep(_node_num, false);
            std::vector<bool> visited(_node_num, false);

            for (int u = 0; u < _node_num; ++u) {
                if (visited[u]) continue;
                is_rep[u] = true;
                
                _parent[u] = _root;
                _pred[u] = _arc_num + u;
                _forward[u] = (_supply[u] >= 0);  // same as init()
                _state[_arc_num + u] = STATE_TREE;
                visited[u] = true;

                std::queue<int> bfs_queue;
                bfs_queue.push(u);
                while (!bfs_queue.empty()) {
                    int v = bfs_queue.front();
                    bfs_queue.pop();
                    for (int k = tree_adj_start[v]; k < tree_adj_start[v + 1]; ++k) {
                        int w = tree_adj_node[k];
                        ArcsType arc_e = tree_adj_arc[k];
                        if (visited[w]) continue;
                        visited[w] = true;
                        
                        _parent[w] = v;
                        _pred[w] = arc_e;
                        _state[arc_e] = STATE_TREE;
                        _forward[w] = (_source[arc_e] == w);
                        
                        _state[_arc_num + w] = STATE_LOWER;
                        _flow[_arc_num + w] = 0;
                        
                        bfs_queue.push(w);
                    }
                }
            }

            // STEP 4: Build thread (preorder traversal)
            {
                std::vector<std::vector<int>> children(_node_num + 1);
                for (int u = 0; u < _node_num; ++u) {
                    children[_parent[u]].push_back(u);
                }

                std::vector<int> preorder;
                preorder.reserve(_node_num + 1);
                std::stack<int> dfs_stack;
                dfs_stack.push(_root);
                while (!dfs_stack.empty()) {
                    int v = dfs_stack.top();
                    dfs_stack.pop();
                    preorder.push_back(v);
                    for (int i = (int)children[v].size() - 1; i >= 0; --i) {
                        dfs_stack.push(children[v][i]);
                    }
                }

                for (int i = 0; i < (int)preorder.size() - 1; ++i) {
                    _thread[preorder[i]] = preorder[i + 1];
                }
                _thread[preorder.back()] = preorder[0];

                for (int u = 0; u <= _node_num; ++u) {
                    _rev_thread[_thread[u]] = u;
                }

                for (int u = 0; u <= _node_num; ++u) {
                    _succ_num[u] = 1;
                }
                for (int i = (int)preorder.size() - 1; i > 0; --i) {
                    int u = preorder[i];
                    _succ_num[_parent[u]] += _succ_num[u];
                }

                std::vector<int> pos(_node_num + 1);
                for (int i = 0; i < (int)preorder.size(); ++i) {
                    pos[preorder[i]] = i;
                }
                for (int i = 0; i < (int)preorder.size(); ++i) {
                    int u = preorder[i];
                    _last_succ[u] = preorder[pos[u] + _succ_num[u] - 1];
                }
            }

            // STEP 5: Compute flows on tree arcs
            {
                std::vector<Value> net(_node_num + 1);
                for (int u = 0; u <= _node_num; ++u) {
                    net[u] = _supply[u];
                }
                
                std::vector<int> preorder;
                preorder.reserve(_node_num + 1);
                int cur = _root;
                for (int i = 0; i <= _node_num; ++i) {
                    preorder.push_back(cur);
                    cur = _thread[cur];
                }
                
                int ejected = 0;
                for (int i = (int)preorder.size() - 1; i > 0; --i) {
                    int u = preorder[i];
                    ArcsType e = _pred[u];
                    
                    Value f = _forward[u] ? net[u] : -net[u];
                    
                    if (f >= 0) {
                        _flow[e] = f;
                        net[_parent[u]] += net[u];
                    } else {
                        if (e < _arc_num) {
                            _state[e] = STATE_LOWER;
                            _flow[e] = 0;
                        }
                        // Reconnect u to root via artificial arc
                        ArcsType art_e = _arc_num + u;
                        _parent[u] = _root;
                        _pred[u] = art_e;
                        _forward[u] = (_source[art_e] == u);
                        _state[art_e] = STATE_TREE;
                        
                        Value art_f = _forward[u] ? net[u] : -net[u];
                        _flow[art_e] = art_f >= 0 ? art_f : -art_f;
                        if (art_f < 0) {
                            _forward[u] = !_forward[u];
                            _flow[art_e] = -art_f;
                        }
                        
                        net[_root] += net[u];
                        ejected++;
                    }
                }
                if (ejected > 0) {
                    std::vector<std::vector<int>> children2(_node_num + 1);
                    for (int u = 0; u < _node_num; ++u) {
                        children2[_parent[u]].push_back(u);
                    }
                    // DFS preorder
                    std::vector<int> preorder2;
                    preorder2.reserve(_node_num + 1);
                    std::stack<int> dfs2;
                    dfs2.push(_root);
                    while (!dfs2.empty()) {
                        int v = dfs2.top(); dfs2.pop();
                        preorder2.push_back(v);
                        for (int j = (int)children2[v].size() - 1; j >= 0; --j) {
                            dfs2.push(children2[v][j]);
                        }
                    }
                    for (int i = 0; i < (int)preorder2.size() - 1; ++i) {
                        _thread[preorder2[i]] = preorder2[i + 1];
                    }
                    _thread[preorder2.back()] = preorder2[0];
                    for (int u = 0; u <= _node_num; ++u) {
                        _rev_thread[_thread[u]] = u;
                    }
                    for (int u = 0; u <= _node_num; ++u) _succ_num[u] = 1;
                    for (int i = (int)preorder2.size() - 1; i > 0; --i) {
                        _succ_num[_parent[preorder2[i]]] += _succ_num[preorder2[i]];
                    }
                    std::vector<int> pos2(_node_num + 1);
                    for (int i = 0; i < (int)preorder2.size(); ++i) pos2[preorder2[i]] = i;
                    for (int i = 0; i < (int)preorder2.size(); ++i) {
                        int u = preorder2[i];
                        _last_succ[u] = preorder2[pos2[u] + _succ_num[u] - 1];
                    }
                }
            }

            // STEP 6: Compute potentials from the final tree
            {
                _pi[_root] = 0;
                int u = _thread[_root];
                while (u != _root) {
                    ArcsType e = _pred[u];
                    int v = _parent[u];
                    if (_forward[u]) {
                        _pi[u] = _pi[v] - getCostForArc(e);
                    } else {
                        _pi[u] = _pi[v] + getCostForArc(e);
                    }
                    u = _thread[u];
                }
            }

            // Initialize in_arc to a valid value
            in_arc = 0;

            return true;
        }

        // Initialize internal data structures
        bool init() {
            if (_node_num == 0) return false;

            // Check the sum of supply values
            _sum_supply = 0;
            for (int i = 0; i != _node_num; ++i) {
                _sum_supply += _supply[i];
            }
            if ( fabs(_sum_supply) > _EPSILON ) return false;

			_sum_supply = 0;

            // Initialize artifical cost
            Cost ART_COST;
            if (std::numeric_limits<Cost>::is_exact) {
                ART_COST = std::numeric_limits<Cost>::max() / 2 + 1;
            } else {
                ART_COST = 0;
                for (ArcsType i = 0; i != _arc_num; ++i) {
                    Cost c = getCostForArc(i);
                    if (c > ART_COST) ART_COST = c;
                }
                ART_COST = (ART_COST + 1) * _node_num;
            }

            memset(&_state[0], STATE_LOWER, _arc_num);

            // Set data for the artificial root node
            _root = _node_num;
            _parent[_root] = -1;
            _pred[_root] = -1;
            _thread[_root] = 0;
            _rev_thread[0] = _root;
            _succ_num[_root] = _node_num + 1;
            _last_succ[_root] = _root - 1;
            _supply[_root] = -_sum_supply;
            _pi[_root] = 0;

            // Add artificial arcs and initialize the spanning tree data structure
            if (_sum_supply == 0) {
                // EQ supply constraints
                _search_arc_num = _arc_num;
                _all_arc_num = _arc_num + _node_num;
                for (ArcsType u = 0, e = _arc_num; u != _node_num; ++u, ++e) {
                    _parent[u] = _root;
                    _pred[u] = e;
                    _thread[u] = u + 1;
                    _rev_thread[u + 1] = u;
                    _succ_num[u] = 1;
                    _last_succ[u] = u;
                    _state[e] = STATE_TREE;
                    if (_supply[u] >= 0) {
                        _forward[u] = true;
                        _pi[u] = 0;
                        _source[e] = u;
                        _target[e] = _root;
                        _flow[e] = _supply[u];
                        _cost[e] = 0;
                    } else {
                        _forward[u] = false;
                        _pi[u] = ART_COST;
                        _source[e] = _root;
                        _target[e] = u;
                        _flow[e] = -_supply[u];
                        _cost[e] = ART_COST;
                    }
                }
            }
            else if (_sum_supply > 0) {
                // LEQ supply constraints
                _search_arc_num = _arc_num + _node_num;
                ArcsType f = _arc_num + _node_num;
                for (ArcsType u = 0, e = _arc_num; u != _node_num; ++u, ++e) {
                    _parent[u] = _root;
                    _thread[u] = u + 1;
                    _rev_thread[u + 1] = u;
                    _succ_num[u] = 1;
                    _last_succ[u] = u;
                    if (_supply[u] >= 0) {
                        _forward[u] = true;
                        _pi[u] = 0;
                        _pred[u] = e;
                        _source[e] = u;
                        _target[e] = _root;
                        _flow[e] = _supply[u];
                        _cost[e] = 0;
                        _state[e] = STATE_TREE;
                    } else {
                        _forward[u] = false;
                        _pi[u] = ART_COST;
                        _pred[u] = f;
                        _source[f] = _root;
                        _target[f] = u;
                        _flow[f] = -_supply[u];
                        _cost[f] = ART_COST;
                        _state[f] = STATE_TREE;
                        _source[e] = u;
                        _target[e] = _root;
                        //_flow[e] = 0;  //by default, the sparse matrix is empty
                        _cost[e] = 0;
                        _state[e] = STATE_LOWER;
                        ++f;
                    }
                }
                _all_arc_num = f;
            }
            else {
                // GEQ supply constraints
                _search_arc_num = _arc_num + _node_num;
                ArcsType f = _arc_num + _node_num;
                for (ArcsType u = 0, e = _arc_num; u != _node_num; ++u, ++e) {
                    _parent[u] = _root;
                    _thread[u] = u + 1;
                    _rev_thread[u + 1] = u;
                    _succ_num[u] = 1;
                    _last_succ[u] = u;
                    if (_supply[u] <= 0) {
                        _forward[u] = false;
                        _pi[u] = 0;
                        _pred[u] = e;
                        _source[e] = _root;
                        _target[e] = u;
                        _flow[e] = -_supply[u];
                        _cost[e] = 0;
                        _state[e] = STATE_TREE;
                    } else {
                        _forward[u] = true;
                        _pi[u] = -ART_COST;
                        _pred[u] = f;
                        _source[f] = u;
                        _target[f] = _root;
                        _flow[f] = _supply[u];
                        _state[f] = STATE_TREE;
                        _cost[f] = ART_COST;
                        _source[e] = _root;
                        _target[e] = u;
                        //_flow[e] = 0;
                        _cost[e] = 0;
                        _state[e] = STATE_LOWER;
                        ++f;
                    }
                }
                _all_arc_num = f;
            }

            return true;
        }

        // Find the join node
        void findJoinNode() {
            int u = _source[in_arc];
            int v = _target[in_arc];
            while (u != v) {
                if (_succ_num[u] < _succ_num[v]) {
                    u = _parent[u];
                } else {
                    v = _parent[v];
                }
            }
            join = u;
        }

        // Find the leaving arc of the cycle and returns true if the
        // leaving arc is not the same as the entering arc
        bool findLeavingArc() {
            // Initialize first and second nodes according to the direction
            // of the cycle
            if (_state[in_arc] == STATE_LOWER) {
                first  = _source[in_arc];
                second = _target[in_arc];
            } else {
                first  = _target[in_arc];
                second = _source[in_arc];
            }
            delta = INF;
            char result = 0;
            Value d;
            ArcsType e;

            // Search the cycle along the path form the first node to the root
            for (int u = first; u != join; u = _parent[u]) {
                e = _pred[u];
                d = _forward[u] ? _flow[e] : INF ;
                if (d < delta) {
                    delta = d;
                    u_out = u;
                    result = 1;
                }
            }
            // Search the cycle along the path form the second node to the root
            for (int u = second; u != join; u = _parent[u]) {
                e = _pred[u];
                d = _forward[u] ? INF  : _flow[e];
                if (d <= delta) {
                    delta = d;
                    u_out = u;
                    result = 2;
                }
            }

            if (result == 1) {
                u_in = first;
                v_in = second;
            } else {
                u_in = second;
                v_in = first;
            }
            return result != 0;
        }

        // Change _flow and _state vectors
        void changeFlow(bool change) {
            // Augment along the cycle
            if (delta > 0) {
                Value val = _state[in_arc] * delta;
                _flow[in_arc] += val;
                for (int u = _source[in_arc]; u != join; u = _parent[u]) {
                    _flow[_pred[u]] += _forward[u] ? -val : val;
                }
                for (int u = _target[in_arc]; u != join; u = _parent[u]) {
                    _flow[_pred[u]] += _forward[u] ? val : -val;
                }
            }
            // Update the state of the entering and leaving arcs
            if (change) {
                _state[in_arc] = STATE_TREE;
                _state[_pred[u_out]] =
                (_flow[_pred[u_out]] == 0) ? STATE_LOWER : STATE_UPPER;
            } else {
                _state[in_arc] = -_state[in_arc];
            }
        }

        // Update the tree structure
        void updateTreeStructure() {
            int u, w;
            int old_rev_thread = _rev_thread[u_out];
            int old_succ_num = _succ_num[u_out];
            int old_last_succ = _last_succ[u_out];
            v_out = _parent[u_out];

            u = _last_succ[u_in];  // the last successor of u_in
            right = _thread[u];    // the node after it

            // Handle the case when old_rev_thread equals to v_in
            // (it also means that join and v_out coincide)
            if (old_rev_thread == v_in) {
                last = _thread[_last_succ[u_out]];
            } else {
                last = _thread[v_in];
            }

            // Update _thread and _parent along the stem nodes (i.e. the nodes
            // between u_in and u_out, whose parent have to be changed)
            _thread[v_in] = stem = u_in;
            _dirty_revs.clear();
            _dirty_revs.push_back(v_in);
            par_stem = v_in;
            while (stem != u_out) {
                // Insert the next stem node into the thread list
                new_stem = _parent[stem];
                _thread[u] = new_stem;
                _dirty_revs.push_back(u);

                // Remove the subtree of stem from the thread list
                w = _rev_thread[stem];
                _thread[w] = right;
                _rev_thread[right] = w;

                // Change the parent node and shift stem nodes
                _parent[stem] = par_stem;
                par_stem = stem;
                stem = new_stem;

                // Update u and right
                u = _last_succ[stem] == _last_succ[par_stem] ?
                _rev_thread[par_stem] : _last_succ[stem];
                right = _thread[u];
            }
            _parent[u_out] = par_stem;
            _thread[u] = last;
            _rev_thread[last] = u;
            _last_succ[u_out] = u;

            // Remove the subtree of u_out from the thread list except for
            // the case when old_rev_thread equals to v_in
            // (it also means that join and v_out coincide)
            if (old_rev_thread != v_in) {
                _thread[old_rev_thread] = right;
                _rev_thread[right] = old_rev_thread;
            }

            // Update _rev_thread using the new _thread values
            for (int i = 0; i != int(_dirty_revs.size()); ++i) {
                int u = _dirty_revs[i];
                _rev_thread[_thread[u]] = u;
            }

            // Update _pred, _forward, _last_succ and _succ_num for the
            // stem nodes from u_out to u_in
            int tmp_sc = 0, tmp_ls = _last_succ[u_out];
            u = u_out;
            while (u != u_in) {
                w = _parent[u];
                _pred[u] = _pred[w];
                _forward[u] = !_forward[w];
                tmp_sc += _succ_num[u] - _succ_num[w];
                _succ_num[u] = tmp_sc;
                _last_succ[w] = tmp_ls;
                u = w;
            }
            _pred[u_in] = in_arc;
            _forward[u_in] = (u_in == _source[in_arc]);
            _succ_num[u_in] = old_succ_num;

            // Set limits for updating _last_succ form v_in and v_out
            // towards the root
            int up_limit_in = -1;
            int up_limit_out = -1;
            if (_last_succ[join] == v_in) {
                up_limit_out = join;
            } else {
                up_limit_in = join;
            }

            // Update _last_succ from v_in towards the root
            for (u = v_in; u != up_limit_in && _last_succ[u] == v_in;
                 u = _parent[u]) {
                _last_succ[u] = _last_succ[u_out];
            }
            // Update _last_succ from v_out towards the root
            if (join != old_rev_thread && v_in != old_rev_thread) {
                for (u = v_out; u != up_limit_out && _last_succ[u] == old_last_succ;
                     u = _parent[u]) {
                    _last_succ[u] = old_rev_thread;
                }
            } else {
                for (u = v_out; u != up_limit_out && _last_succ[u] == old_last_succ;
                     u = _parent[u]) {
                    _last_succ[u] = _last_succ[u_out];
                }
            }

            // Update _succ_num from v_in to join
            for (u = v_in; u != join; u = _parent[u]) {
                _succ_num[u] += old_succ_num;
            }
            // Update _succ_num from v_out to join
            for (u = v_out; u != join; u = _parent[u]) {
                _succ_num[u] -= old_succ_num;
            }
        }

        // Update potentials
        void updatePotential() {
            Cost sigma = _forward[u_in] ?
            _pi[v_in] - _pi[u_in] - getCostForArc(_pred[u_in]) :
            _pi[v_in] - _pi[u_in] + getCostForArc(_pred[u_in]);
            // Update potentials in the subtree, which has been moved
            int end = _thread[_last_succ[u_in]];
            for (int u = u_in; u != end; u = _thread[u]) {
                _pi[u] += sigma;
            }
        }

        // Heuristic initial pivots
        bool initialPivots() {
            Value curr, total = 0;
            std::vector<Node> supply_nodes, demand_nodes;
            Node u; _graph.first(u);
            for (; u != INVALIDNODE; _graph.next(u)) {
                curr = _supply[_node_id(u)];
                if (curr > 0) {
                    total += curr;
                    supply_nodes.push_back(u);
                }
                else if (curr < 0) {
                    demand_nodes.push_back(u);
                }
            }
            if (_sum_supply > 0) total -= _sum_supply;
            if (total <= 0) return true;

            ArcVector arc_vector;
            if (_sum_supply >= 0) {
                if (supply_nodes.size() == 1 && demand_nodes.size() == 1) {
                    // Perform a reverse graph search from the sink to the source
                    BoolVector reached(_node_num, false);
                    Node s = supply_nodes[0], t = demand_nodes[0];
                    std::vector<Node> stack;
                    reached[t] = true;
                    stack.push_back(t);
                    while (!stack.empty()) {
                        Node u, v = stack.back();
                        stack.pop_back();
                        if (v == s) break;
                        Arc a; _graph.firstIn(a, v);
                        for (; a != INVALID; _graph.nextIn(a)) {
                            if (reached[u = _graph.source(a)]) continue;
                            ArcsType j = getArcID(a);
                            if (INF >= total) {
                                arc_vector.push_back(j);
                                reached[u] = true;
                                stack.push_back(u);
                            }
                        }
                    }
                } else {
                    // Find the min. cost incomming arc for each demand node
                    for (int i = 0; i != demand_nodes.size(); ++i) {
                        Node v = demand_nodes[i];
                        Cost c, min_cost = std::numeric_limits<Cost>::max();
                        Arc min_arc = INVALID;
                        Arc a; _graph.firstIn(a, v);
                        for (; a != INVALID; _graph.nextIn(a)) {
                            c = getCostForArc(getArcID(a));
                            if (c < min_cost) {
                                min_cost = c;
                                min_arc = a;
                            }
                        }
                        if (min_arc != INVALID) {
                            arc_vector.push_back(getArcID(min_arc));
                        }
                    }
                }
            } else {
                // Find the min. cost outgoing arc for each supply node
                for (int i = 0; i != int(supply_nodes.size()); ++i) {
                    Node u = supply_nodes[i];
                    Cost c, min_cost = std::numeric_limits<Cost>::max();
                    Arc min_arc = INVALID;
                    Arc a; _graph.firstOut(a, u);
                    for (; a != INVALID; _graph.nextOut(a)) {
                        c = getCostForArc(getArcID(a));
                        if (c < min_cost) {
                            min_cost = c;
                            min_arc = a;
                        }
                    }
                    if (min_arc != INVALID) {
                        arc_vector.push_back(getArcID(min_arc));
                    }
                }
            }

            // Perform heuristic initial pivots
            for (ArcsType i = 0; i != arc_vector.size(); ++i) {
                in_arc = arc_vector[i];
                // l'erreur est probablement ici...
                if (_state[in_arc] * (getCostForArc(in_arc) + _pi[_source[in_arc]] -
                                      _pi[_target[in_arc]]) >= 0) continue;
                findJoinNode();
                bool change = findLeavingArc();
                if (delta >= MAX) return false;
                changeFlow(change);
                if (change) {
                    updateTreeStructure();
                    updatePotential();
                }
            }
            return true;
        }

        // Execute the algorithm
        ProblemType start() {
            return start<BlockSearchPivotRule>();
        }

        template <typename PivotRuleImpl>
        ProblemType start() {
            PivotRuleImpl pivot(*this);
			ProblemType retVal = OPTIMAL;

            // Perform heuristic initial pivots (skip if warmstart tree was built)
            if (!_warmstart_tree_built) {
                if (!initialPivots()) return UNBOUNDED;
            }

            uint64_t iter_number = 0;
            //pivot.setDantzig(true);
            // Execute the Network Simplex algorithm
            while (pivot.findEnteringArc()) {
                if(max_iter > 0 && ++iter_number>=max_iter&&max_iter>0){
                    // max iterations hit
					retVal = MAX_ITER_REACHED;
                    break;
                }

                findJoinNode();
                bool change = findLeavingArc();
                if (delta >= MAX) return UNBOUNDED;
                changeFlow(change);
                if (change) {
                    updateTreeStructure();
                    updatePotential();
                }

            }

            // Check feasibility
			if( retVal == OPTIMAL){
                for (ArcsType e = _search_arc_num; e != _all_arc_num; ++e) {
                    if (_flow[e] != 0){
                        if (fabs(_flow[e]) > _EPSILON) // change of the original code following issue #126
                            return INFEASIBLE;
                        else
                            _flow[e]=0;

                    }
                }
			}

            // Shift potentials to meet the requirements of the GEQ/LEQ type
            // optimality conditions
            if (_sum_supply == 0) {
                if (_stype == GEQ) {
                    Cost max_pot = -std::numeric_limits<Cost>::max();
                    for (ArcsType i = 0; i != _node_num; ++i) {
                        if (_pi[i] > max_pot) max_pot = _pi[i];
                    }
                    if (max_pot > 0) {
                        for (ArcsType i = 0; i != _node_num; ++i)
                            _pi[i] -= max_pot;
                    }
                } else {
                    Cost min_pot = std::numeric_limits<Cost>::max();
                    for (ArcsType i = 0; i != _node_num; ++i) {
                        if (_pi[i] < min_pot) min_pot = _pi[i];
                    }
                    if (min_pot < 0) {
                        for (ArcsType i = 0; i != _node_num; ++i)
                            _pi[i] -= min_pot;
                    }
                }
            }

            return retVal;
        }

    }; //class NetworkSimplexSimple

    ///@}

} //namespace lemon
