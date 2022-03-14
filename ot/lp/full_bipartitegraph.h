/* -*- mode: C++; indent-tabs-mode: nil; -*-
 *
 * This file has been adapted by Nicolas Bonneel (2013), 
 * from full_graph.h from LEMON, a generic C++ optimization library,
 * to implement a lightweight fully connected bipartite graph. A previous
 * version of this file is used as part of the Displacement Interpolation 
 * project, 
 * Web: http://www.cs.ubc.ca/labs/imager/tr/2011/DisplacementInterpolation/
 * 
 *
 **** Original file Copyright Notice :
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

#include "core.h"
#include <cstdint>

///\ingroup graphs
///\file
///\brief FullBipartiteDigraph and FullBipartiteGraph classes.


namespace lemon {


  class FullBipartiteDigraphBase {
  public:

    typedef FullBipartiteDigraphBase Digraph;

    //class Node;
	typedef int Node;
    //class Arc;
	typedef int64_t Arc;

  protected:

    int _node_num;
    int64_t _arc_num;
	
    FullBipartiteDigraphBase() {}

    void construct(int n1, int n2) { _node_num = n1+n2; _arc_num = (int64_t)n1 * (int64_t)n2; _n1=n1; _n2=n2;}

  public:

	int _n1, _n2;


    Node operator()(int ix) const { return Node(ix); }
    static int index(const Node& node) { return node; }

    Arc arc(const Node& s, const Node& t) const {
		if (s<_n1 && t>=_n1)
            return Arc((int64_t)s * (int64_t)_n2 + (int64_t)(t-_n1) );
		else
			return Arc(-1);
    }

    int nodeNum() const { return _node_num; }
    int64_t arcNum() const { return _arc_num; }

    int maxNodeId() const { return _node_num - 1; }
    int64_t maxArcId() const { return _arc_num - 1; }

    Node source(Arc arc) const { return arc / _n2; }
    Node target(Arc arc) const { return (arc % _n2) + _n1; }

    static int id(Node node) { return node; }
    static int64_t id(Arc arc) { return arc; }

    static Node nodeFromId(int id) { return Node(id);}
    static Arc arcFromId(int64_t id) { return Arc(id);}


    Arc findArc(Node s, Node t, Arc prev = -1) const {
      return prev == -1 ? arc(s, t) : -1;
    }

    void first(Node& node) const {
      node = _node_num - 1;
    }

    static void next(Node& node) {
      --node;
    }

    void first(Arc& arc) const {
      arc = _arc_num - 1;
    }

    static void next(Arc& arc) {
      --arc;
    }

    void firstOut(Arc& arc, const Node& node) const {
		if (node>=_n1)
			arc = -1;
		else
			arc = (node + 1) * _n2 - 1;
    }

    void nextOut(Arc& arc) const {
      if (arc % _n2 == 0) arc = 0;
      --arc;
    }

    void firstIn(Arc& arc, const Node& node) const {
		if (node<_n1)
			arc = -1;
		else
			arc = _arc_num + node - _node_num;
    }

    void nextIn(Arc& arc) const {
      arc -= _n2;
      if (arc < 0) arc = -1;
    }

  };

  /// \ingroup graphs
  ///
  /// \brief A directed full graph class.
  ///
  /// FullBipartiteDigraph is a simple and fast implementation of directed full
  /// (complete) graphs. It contains an arc from each node to each node
  /// (including a loop for each node), therefore the number of arcs
  /// is the square of the number of nodes.
  /// This class is completely static and it needs constant memory space.
  /// Thus you can neither add nor delete nodes or arcs, however
  /// the structure can be resized using resize().
  ///
  /// This type fully conforms to the \ref concepts::Digraph "Digraph concept".
  /// Most of its member functions and nested classes are documented
  /// only in the concept class.
  ///
  /// This class provides constant time counting for nodes and arcs.
  ///
  /// \note FullBipartiteDigraph and FullBipartiteGraph classes are very similar,
  /// but there are two differences. While this class conforms only
  /// to the \ref concepts::Digraph "Digraph" concept, FullBipartiteGraph
  /// conforms to the \ref concepts::Graph "Graph" concept,
  /// moreover FullBipartiteGraph does not contain a loop for each
  /// node as this class does.
  ///
  /// \sa FullBipartiteGraph
  class FullBipartiteDigraph : public FullBipartiteDigraphBase {
    typedef FullBipartiteDigraphBase Parent;

  public:

    /// \brief Default constructor.
    ///
    /// Default constructor. The number of nodes and arcs will be zero.
    FullBipartiteDigraph() { construct(0,0); }

    /// \brief Constructor
    ///
    /// Constructor.
    /// \param n The number of the nodes.
    FullBipartiteDigraph(int n1, int n2) { construct(n1, n2); }


    /// \brief Returns the node with the given index.
    ///
    /// Returns the node with the given index. Since this structure is
    /// completely static, the nodes can be indexed with integers from
    /// the range <tt>[0..nodeNum()-1]</tt>.
    /// The index of a node is the same as its ID.
    /// \sa index()
    Node operator()(int ix) const { return Parent::operator()(ix); }

    /// \brief Returns the index of the given node.
    ///
    /// Returns the index of the given node. Since this structure is
    /// completely static, the nodes can be indexed with integers from
    /// the range <tt>[0..nodeNum()-1]</tt>.
    /// The index of a node is the same as its ID.
    /// \sa operator()()
    static int index(const Node& node) { return Parent::index(node); }

    /// \brief Returns the arc connecting the given nodes.
    ///
    /// Returns the arc connecting the given nodes.
    /*Arc arc(Node u, Node v) const {
      return Parent::arc(u, v);
    }*/

    /// \brief Number of nodes.
    int nodeNum() const { return Parent::nodeNum(); }
    /// \brief Number of arcs.
    int64_t arcNum() const { return Parent::arcNum(); }
  };




} //namespace lemon
