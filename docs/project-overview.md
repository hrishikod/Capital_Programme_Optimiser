---
title: Capital Programme Portfolio Optimization

author: Hrishi Kodthuguli
---

# Capital Programme Portfolio Optimization

## Product

This page helps newcomers navigate quickly — used by everyone.

### Overview (1–2 sentences)

<Insert a short description of the output the optimisation model generates>

### Services & Service Quality

<What does the product provide? Frequency, reliability, expected accuracy/quality>

### Product Owner

<Name>

### Document Repository

<Link>

### Code Repository

<Link to Databricks repo, GitHub, or both>

### Jira Board

<Link>

### Output Data Location

<e.g., Unity Catalogue path, S3/ADLS path, or Databricks tables>

### Jargon

<List of key terms used in the project (e.g., benefit scores, discount rate, dependencies, constraints)>

---

## Discovery and Brief

The Capital Programme Portfolio Optimisation project was created to help schedule and sequence major GPS27 capital projects so that total Net Present Value benefits are maximised, while staying within yearly and total funding limits.

Previously, portfolio decisions were made using manual scenario testing and spreadsheets, which made it hard to ensure consistency, transparency, and the best use of limited investment funding.

The optimisation model consolidates all key information—project costs, expected benefits, timing, and funding constraints—enabling decisions to be made in a more transparent and evidence-based manner. The model examines all projects and selects the combination and timing that yields the greatest value within the available budget.

The latest version of the tool includes features such as:

- setting yearly funding limits with optional flexibility

- automatically identifying project start times that won’t fit the budget

- options to include, exclude, or fix the start year of certain projects

- a faster solving method, so results return quickly

- the ability to consider multiple types of benefits

- optional checks to keep total spending within a fixed amount

- automatic generation of schedules, spending profiles, and benefit summaries

---

## Description

Background context — used by Benefactors.

### Objectives

<What the project aims to achieve>

### Value Proposition

<Why this work matters — e.g., optimisation creates X% efficiency, reduces budget overruns, prioritises projects, etc.>

### Key Deliverables

- Initial optimisation prototype

- Productionised optimisation engine

- Reporting/dashboard outputs

- Scenario comparison tool

- Documentation & knowledge transfer

### Key People

- Product Owner

- Data Scientist(s)

- Business stakeholder(s)

- Technical stakeholder(s)

---
## Project Management

Project governance — used by Benefactors.

### Team

<List of team members and roles>

### Milestones, Key Dates & Changes

<Table of key dates>

### Risk Register

Link to PIE Risk Register + project-specific view

### Stakeholder Engagement

- Business consults

- Technical consults

- Steering/governance updates

### Sprint Review Learnings

<Notes or summary of key learnings>

### RASCI / RAPID

<Decision-making matrix>

---
## Architecture

Tooling requirements — used by Digital.

### Data Dependencies

<Source systems, access methods, update frequency, tables involved> 

### Infrastructure

<Clusters catalogues, job workflows, storage, permissions>

### Dataflow

<Diagram or step-by-step process>

### Output Schema

<List and describe all output fields>

---
## User Guide

Helps end users interpret the outputs — used by end users.

- How to run or access outputs
- How to read optimisation results
- Example use cases (e.g., budget scenarios, risk-based prioritisation)

---
## Model Logic / Components

Detailed documentation — used by the Product team.

### Current State

<Description of the model as of today>

### Assumptions

- Budget constraints
- Project interdependencies
- Objective function assumptions
- Data quality assumptions
- (Or whatever applies)

### Limitations / Constraints

- Data gaps
- Solver performance
- Known issues

### Business Logic

Detailed breakdown of transformation rules, scoring methods, prioritisation logic, constraints, etc.

### Proposed Solutions (Tracked with Status)

A list of improvements, backlog items, future enhancements.

---	
## Reports

Highlights released work — used by end users.

- Output datasets
- Dashboards
- Scenario analysis reports
- Insights generated over time
