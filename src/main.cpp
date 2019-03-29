#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdio>
#include <exception>

#include <Eigen/Dense>
#include <gurobi_c++.h>


using Eigen::ArrayXXi;
using Eigen::Dynamic;
using Eigen::Array;

char varname[256];

int main1(int argn, char** argv)
{
  std::ifstream in;
  in.open(argv[1]);

  std::cout << "loading file:" << argv[1] << std::endl;

  int n_row;
  int n_col;
  int n_vehic;
  int n_rides;
  int bonus_point;
  int n_turns;

  {
    std::string _line;
    std::getline(in,_line);
    std::stringstream header(_line);
    header >> n_row;
    header >> n_col;
    header >> n_vehic;
    header >> n_rides;
    header >> bonus_point;
    header >> n_turns;
  }
  ArrayXXi start_pos = ArrayXXi::Zero(n_rides,2);
  ArrayXXi end_pos = ArrayXXi::Zero(n_rides,2);
  ArrayXXi early_start_turn = ArrayXXi::Zero(n_rides,1);
  ArrayXXi late_end_turn = ArrayXXi::Zero(n_rides,1);

  for (int i=0;i<n_rides;i++) {
    std::string _line;
    std::getline(in,_line);
    std::stringstream line(_line);

    line >> start_pos(i,0);
    line >> start_pos(i,1);
    line >> end_pos(i,0);
    line >> end_pos(i,1);
    line >> early_start_turn(i);
    line >> late_end_turn(i);
  }

  ArrayXXi length_trip = (start_pos - end_pos).abs().rowwise().sum();

  ArrayXXi late_start_turn = late_end_turn - length_trip;
  ArrayXXi early_end_turn = early_start_turn + length_trip;
  ArrayXXi max_delay = late_start_turn - early_start_turn;

  ArrayXXi start_distance = ArrayXXi::Zero(n_rides,2);
  ArrayXXi distance = ArrayXXi::Zero(n_rides,n_rides);
  Array<bool,Dynamic,Dynamic> valid_transition = Array<bool,Dynamic,Dynamic>::Zero(n_rides,n_rides);

  for (int i=0;i<n_rides;i++) {
    start_distance(i) = start_pos(i,0) + start_pos(i,1);

    for (int j=0;j<n_rides;j++) {
      distance(i,j) = (end_pos.row(i) - start_pos.row(j)).abs().sum();

      if (early_end_turn(i) + distance(i,j) <= late_start_turn(j) && i != j) {
          valid_transition(i,j) = true;
      }
    }
  }

  GRBEnv env = GRBEnv();
  GRBModel m = GRBModel(env);

  Array<GRBVar,Dynamic,Dynamic> start_transition(n_rides,1);
  Array<GRBVar,Dynamic,Dynamic> end_transition(n_rides,1);
  Array<GRBVar,Dynamic,Dynamic> transition(n_rides,n_rides);
  Array<GRBVar,Dynamic,Dynamic> delay(n_rides,1);
  Array<GRBVar,Dynamic,Dynamic> bonus(n_rides,1);


  for (int i=0;i<n_rides;i++) {

    sprintf(varname,"t_S_%d",i);
    start_transition(i) = m.addVar(0.0,1.0,0.0,GRB_BINARY,varname);

    sprintf(varname,"t_%d_E",i);
    end_transition(i) = m.addVar(0.0,1.0,0.0,GRB_BINARY,varname);

    sprintf(varname,"d_%d",i);
    delay(i) = m.addVar(0.0,max_delay(i),0.0,GRB_INTEGER,varname);

    sprintf(varname,"bonus_%d",i);
    bonus(i) = m.addVar(0.0,1.0,0.0,GRB_BINARY,varname);

    for (int j=0;j<n_rides;j++) {
      if (valid_transition(i,j)) {
        sprintf(varname,"t_%d_%d",i,j);
        transition(i,j) = m.addVar(0.0,1.0,0.0,GRB_BINARY,varname);
      }
    }
  }

  {
    GRBLinExpr start = 0;
    for (int i=0;i<n_rides;i++) {
      start += start_transition(i);
    }
    m.addConstr(start <= n_vehic, "start_constrain");
  }

  for (int i=0;i<n_rides;i++) {
    GRBLinExpr incoming_transitions = 0;
    GRBLinExpr outgoing_transitions = 0;

    incoming_transitions += start_transition(i);
    for (int j=0;j<n_rides;j++) {
      if (valid_transition(j,i)) {
        incoming_transitions += transition(j,i);
      }
    }

    outgoing_transitions += end_transition(i);
    for (int j=0;j<n_rides;j++) {
      if (valid_transition(i,j)) {
        outgoing_transitions += transition(i,j);
      }
    }

    sprintf(varname,"at_most_one_%d",i);
    m.addConstr(incoming_transitions <= 1, varname);

    sprintf(varname,"io_euqal_%d",i);
    m.addConstr(incoming_transitions == outgoing_transitions, varname);

    sprintf(varname,"bonus_if_ride_taken_%d",i);
    m.addConstr(incoming_transitions >= bonus(i), varname);

    sprintf(varname,"bonus_apply_%d",i);
    m.addConstr((1 - bonus(i)) * max_delay(i) >= delay(i),varname);

    sprintf(varname,"delay_S_%d",i);
    m.addConstr(
      early_start_turn(i) +
      delay(i) +
      (1-start_transition(i))*(start_distance(i) -
      early_start_turn(i) -
      max_delay(i))
      >=
      start_distance(i)
      ,varname);

    for (int j=0;j<n_rides;j++) {
      if (valid_transition(i,j)) {
        sprintf(varname,"delay_%d_%d",i,j);
        m.addConstr(
          early_end_turn(i) +
          delay(i) +
          distance(i,j)
          <=
          early_start_turn(j) +
          delay(j) +
          (1-transition(i,j))*
          (max_delay(i) + distance(i,j) + early_end_turn(i) - early_start_turn(j))
          ,varname);
      }
    }
  }


  GRBLinExpr objective = 0;

  for(int i=0;i<n_rides;i++) {
    objective+= bonus_point*bonus(i);
    objective += start_transition(i) * length_trip(i);

    for(int j=0;j<n_rides;j++) {
      if (valid_transition(i,j)) {
        objective+= length_trip(j)*transition(i,j);
      }
    }
  }

  m.setObjective(objective,GRB_MAXIMIZE);
  m.write("model.lp");
  m.optimize();
  m.write("model.sol");
  {
    std::ofstream out;
    out.open("out.txt");
    int j = 0;
    for (int i=0;i<n_vehic;i++) {
      out <<i<<" ";
      for(;start_transition(j).get(GRB_DoubleAttr_X)<1;j++){}
      int node = j;
      j++;
      while (node != n_rides) {
        out <<node<<" ";
        int k=0;
        for(;transition(node,k).get(GRB_DoubleAttr_X)<1 && k<n_rides;k++){}
        node = k;
      }
      out << std::endl;
      std::cout<<i<<std::endl;
    }
  }
}

int main(int argn, char** argv) {
  try {
    main1(argn,argv);
  } catch (GRBException& e) {
    std::cerr << e.getMessage() << std::endl;
  }
}
