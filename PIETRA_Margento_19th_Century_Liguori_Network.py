#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import networkx as nx
import itertools
import numpy as np

excel_file = '19th_century_translations.xlsx'
df = pd.read_excel(excel_file)

G = nx.DiGraph()

# Function to extract and sort years chronologically
def extract_years(row):
    year = str(row['Year'])[:4]
    earlier = str(row['Earlier'])  # Ensure NaN values are converted to string
    subsequent = str(row['Subsequent Listing'])  # Ensure NaN values are converted to string
    
    # Extract years from Earlier column
    earlier_years = []
    if earlier and earlier != 'nan':  # Check for NaN values
        earlier_years = [x.split()[0] for x in earlier.split(',')]

    # Extract years from Subsequent Listing column
    subsequent_years = []
    if subsequent and subsequent != 'nan':  # Check for NaN values
        subsequent_years = [x.split()[0] for x in subsequent.split(',')]

    # Combine all years and sort them chronologically
    all_years = [year] + earlier_years + subsequent_years
    all_years = [int(y.split()[0]) if y.split()[0].isdigit() else None for y in all_years]
    all_years = sorted([y for y in all_years if y is not None])
    
    return all_years

filtered_df = df.iloc[0:179]

unique_publishers = set()
for publishers in filtered_df['Publisher']:
    if isinstance(publishers, str):
        unique_publishers.update([publisher.strip() for publisher in publishers.split(';')])
    else:
        unique_publishers.update([publishers])

titles = []
years = []
editions = []
publishers = []
translators = []
places = []

for index, row in filtered_df.iterrows():
#for row in filtered_df.iterrows():
    titles.append(row['Title'])
    years.append(extract_years(row))
    editions.append(int(row['Number of Editions']) if not np.isnan(row['Number of Editions']) else 1)
    publishers.append([publisher.strip() for publisher in row['Publisher'].split('; ')] if isinstance(row['Publisher'], str) else row['Publisher'].strip())
    translators.append(row['Translator'])
    places.append([place.strip() for place in row['Place of Publication'].split('; ')] if isinstance(row['Place of Publication'], str) else str(row['Place of Publication']).strip())

for publisher in unique_publishers:
        G.add_node(publisher, publications = [])
    
for specific_publisher in G.nodes():
    for title, year, edition, publisher, translator, place in zip(titles, years, editions, publishers, translators, places):
        if specific_publisher in publisher:
            G.nodes[specific_publisher]['publications'].append({'title': title, 'years': year, 'editions': [edition], 'publisher': publisher, 'translator': translator, 'place': place})

for publisher in G.nodes():
    magnitude = sum(sum(pub['editions']) for pub in G.nodes[publisher]['publications'])
    G.nodes[publisher]['magnitude'] = magnitude


# In[165]:


G.nodes["Benziger Brothers"]["publications"]


# In[4]:


G.nodes["Benziger Brothers"]["publications"][0]["years"]


# In[2]:


G.nodes["Benziger Brothers"]["publications"][0]["title"]


# In[11]:


len(G.nodes())


# In[49]:


# Function to add edges with attributes
def add_edge_with_attributes(G, publisher_A, publisher_B, title, source_years, target_years,
                             source_translator, target_translator, source_num_editions,
                             target_num_editions, source_place, target_place, weight, relation, key):
    G.add_edge(publisher_A, publisher_B, title=title, source_years=source_years, target_years=target_years,
               source_translator=source_translator, target_translator=target_translator,
               source_num_editions=source_num_editions, target_num_editions=target_num_editions,
               source_place=source_place, target_place=target_place, weight=weight, relation=relation, key=key)


# In[57]:


def add_edges_between_publishers(G, publisher_A, publisher_B, publication_A, publication_B):
    years_A = publication_A['years']
    years_B = publication_B['years']
    translators_A = publication_A['translator']
    translators_B = publication_B['translator']
    
    if not pd.isna(translators_A) and not pd.isna(translators_B):
                same_translator = ((translators_A == translators_B) or (translators_A in translators_B) or (translators_B in translators_A))
    else:
                same_translator = False
            
    if pd.isna(translators_A) or 'not named' in translators_A or 'not mentioned' in translators_A or 'None' in translators_A:
                same_translator = False
    if pd.isna(translators_B) or 'not named' in translators_B or 'not mentioned' in translators_B or 'None' in translators_B:
                same_translator = False
    
    # Check for copublication relation
    if publication_A['publisher'] == publication_B['publisher'] and publication_A['title'] == publication_B['title'] and set(years_A) == set(years_B):
        weight = max(publication_A['editions'], publication_B['editions'])
        # Add copublication edge with a unique key based on the relation type
        edge_key = f"copub_{publisher_A}_{publisher_B}_{publication_A['title']}"
        add_edge_with_attributes(G, publisher_A, publisher_B, publication_A['title'], 
                                 years_A, years_B, translators_A, translators_B,
                                 publication_A['editions'], publication_B['editions'],
                                 publication_A['place'], publication_B['place'],
                                 weight, 'copublication', key=edge_key)

    # Check for reprint relation
    if translators_A == translators_B and publication_A['title'] == publication_B['title'] and years_A and years_B and years_A[0] < years_B[0] and publication_A['publisher'] != publication_B['publisher']:
        weight = publication_B['editions']
        # Add reprint edge with a unique key based on the relation type
        edge_key = f"reprint_{publisher_A}_{publisher_B}_{publication_A['title']}"
        add_edge_with_attributes(G, publisher_A, publisher_B, publication_A['title'], 
                                 years_A, years_B, translators_A, translators_B,
                                 publication_A['editions'], publication_B['editions'],
                                 publication_A['place'], publication_B['place'],
                                 weight, 'reprint', key=edge_key)

    # Check for retranslation relation
    if translators_A != translators_B and publication_A['title'] == publication_B['title'] and years_A and years_B and years_A[0] < years_B[0] and publication_A['publisher'] != publication_B['publisher']:
        weight = publication_B['editions']
        # Add retranslation edge with a unique key based on the relation type
        edge_key = f"retrans_{publisher_A}_{publisher_B}_{publication_A['title']}"
        add_edge_with_attributes(G, publisher_A, publisher_B, publication_A['title'], 
                                 years_A, years_B, translators_A, translators_B,
                                 publication_A['editions'], publication_B['editions'],
                                 publication_A['place'], publication_B['place'],
                                 weight, 'retranslation', key=edge_key)

# Iterate over combinations of nodes (publishers)
for publisher_A, publisher_B in itertools.combinations(G11.nodes(), 2):
    publications_A = G11.nodes[publisher_A]["publications"]
    publications_B = G11.nodes[publisher_B]["publications"]
    
    # Iterate over publications of each publisher pair
    for publication_A in publications_A:
        for publication_B in publications_B:
            # Add edges based on different relations
            add_edges_between_publishers(G11, publisher_A, publisher_B, publication_A, publication_B)


# In[58]:


len(G.edges()


# In[249]:


sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1].get('magnitude', 0), reverse=True)

# Print nodes in descending order of magnitude
for node, attrs in sorted_nodes:
    print(f"{node}, Magnitude: {attrs.get('magnitude', 0)}")


# In[83]:


#import networkx as nx

out_degrees = dict(G11.out_degree())
in_degrees = dict(G11.in_degree())
overall_degrees = {node: out_degrees.get(node, 0) + in_degrees.get(node, 0) for node in G11.nodes()}

nodes_by_out_degree = sorted(out_degrees, key=out_degrees.get, reverse=True)
nodes_by_in_degree = sorted(in_degrees, key=in_degrees.get, reverse=True)
nodes_by_overall_degree = sorted(overall_degrees, key=overall_degrees.get, reverse=True)

print("Nodes ordered by outgoing degree:")
for node in nodes_by_out_degree:
    print(node, out_degrees[node])

print("\nNodes ordered by incoming degree:")
for node in nodes_by_in_degree:
    print(node, in_degrees[node])

print("\nNodes ordered by overall degree:")
for node in nodes_by_overall_degree:
    print(node, overall_degrees[node])


# In[10]:


outgoing_weighted_degrees = {}
incoming_weighted_degrees = {}
overall_weighted_degrees = {}

for node in G.nodes():
    outgoing_weight = sum(sum(G.edges[node, neighbor]['weight']) for neighbor in G.successors(node))
    outgoing_weighted_degrees[node] = outgoing_weight

for node in G.nodes():
    incoming_weight = sum(sum(G.edges[neighbor, node]['weight']) for neighbor in G.predecessors(node))
    incoming_weighted_degrees[node] = incoming_weight

for node in G.nodes():
    overall_weighted_degree = outgoing_weighted_degrees.get(node, 0) + incoming_weighted_degrees.get(node, 0)
    overall_weighted_degrees[node] = overall_weighted_degree

sorted_outgoing_weighted_degrees = dict(sorted(outgoing_weighted_degrees.items(), key=lambda item: item[1], reverse=True))
sorted_incoming_weighted_degrees = dict(sorted(incoming_weighted_degrees.items(), key=lambda item: item[1], reverse=True))
sorted_overall_weighted_degrees = dict(sorted(overall_weighted_degrees.items(), key=lambda item: item[1], reverse=True))

print("Outgoing Weighted Degrees:", sorted_outgoing_weighted_degrees)
print("Incoming Weighted Degrees:", sorted_incoming_weighted_degrees)
print("Overall Weighted Degrees:", sorted_overall_weighted_degrees)


# In[61]:


outgoing_weighted_degrees = {}
incoming_weighted_degrees = {}
overall_weighted_degrees = {}

for node in G11.nodes():
    outgoing_weight = sum(sum(G11.edges[node, neighbor]['weight']) for neighbor in G11.successors(node))
    outgoing_weighted_degrees[node] = outgoing_weight

for node in G11.nodes():
    incoming_weight = sum(sum(G11.edges[neighbor, node]['weight']) for neighbor in G11.predecessors(node))
    incoming_weighted_degrees[node] = incoming_weight

for node in G11.nodes():
    overall_weighted_degree = outgoing_weighted_degrees.get(node, 0) + incoming_weighted_degrees.get(node, 0)
    overall_weighted_degrees[node] = overall_weighted_degree

sorted_outgoing_weighted_degrees = dict(sorted(outgoing_weighted_degrees.items(), key=lambda item: item[1], reverse=True))
sorted_incoming_weighted_degrees = dict(sorted(incoming_weighted_degrees.items(), key=lambda item: item[1], reverse=True))
sorted_overall_weighted_degrees = dict(sorted(overall_weighted_degrees.items(), key=lambda item: item[1], reverse=True))

print("Outgoing Weighted Degrees:", sorted_outgoing_weighted_degrees)
print("Incoming Weighted Degrees:", sorted_incoming_weighted_degrees)
print("Overall Weighted Degrees:", sorted_overall_weighted_degrees)


# In[64]:


G11.edges("P.J. Kenedy")


# In[65]:


G11.edges(['P.J. Kenedy', 'Benziger Brothers'])


# In[66]:


G11.edges('P.J. Kenedy', 'Benziger Brothers')


# In[67]:


G.get_edge_data('P.J. Kenedy', 'Benziger Brothers')


# In[68]:


G.get_edge_data('P.J. Kenedy', 'Burns and Oates')


# In[74]:


G11.edges('P.J. Kenedy', 'Burns and Oates', keys = True)


# In[81]:


G11.edges('P.J. Kenedy', data = True)


# In[69]:


G11.successors('Benziger Brothers')


# In[70]:


l = [G11.edges['Benziger Brothers', neighbor]['weight'] for neighbor in G11.successors(node)]


# In[76]:


G11.edges(data=True)


# In[82]:


node = 'Benziger Brothers'

for node_id in G11.successors(node):
    print(node_id)


# In[84]:


node1 = 'James Duffy'

for node_id in G11.successors(node1):
    print(node_id)


# In[73]:


node = 'Benziger Brothers'

successors = G11.successors(node)

weights = []

for neighbor in successors:
    #edges = G11.edges(node, neighbor, keys=True)
    edges = G11.edges(node, neighbor, keys = True, data=True)
    #for u, v, key, attr in edges:
    for u, v, key, in edges:
        weight = attr['weight']
        weights.append(weight)

print("Weights of edges:", weights)


# In[87]:


# Nodes of interest
node1 = 'M.H. Gill & Sons'
node2 = 'Benziger Brothers'

# Initialize a list to store weights of edges between node1 and node2
weights = []

# Iterate over all edges in the graph
for u, v, key, attr in G11.edges(keys=True, data=True):
    # Check if the edge is between node1 and node2
    if (u == node1 and v == node2) or (u == node2 and v == node1):
        # Extract weight from the edge attributes
        weight = attr['weight']
        weights.append(weight)

# Now 'weights' list contains all edge weights between node1 and node2
print("Weights of edges between '{}' and '{}':".format(node1, node2), weights)


# In[88]:


node = 'Benziger Brothers'

successors = []

for node_id in G11.successors(node):
    successors.append(node_id)
    
weights = []

# Iterate over all edges in the graph
for u, v, key, attr in G11.edges(keys=True, data=True):
    # Check if the edge is between node1 and node2
    if (u in successors and v == node) or (u == node and v in successors):
        # Extract weight from the edge attributes
        weight = attr['weight']
        weights.append(weight)


# In[89]:


weights


# In[91]:


#node = 'Benziger Brothers'

#successors = []

#for node_id in G11.successors(node):
    #successors.append(node_id)
    
weights = []

# Iterate over all edges in the graph
#for u, v, key, attr in G11.edges(keys=True, data=True):
    # Check if the edge is between node1 and node2
    #if (u in successors and v == node) or (u == node and v in successors):
        # Extract weight from the edge attributes
        #weight = attr['weight']
        #weights.append(weight)

outgoing_weighted_degrees = {}
incoming_weighted_degrees = {}
overall_weighted_degrees = {}

for node in G11.nodes():
    successors = []
    for node_id in G11.successors(node):
        successors.append(node_id)
    weights = []
    for u, v, key, attr in G11.edges(keys=True, data=True):
        if (u in successors and v == node) or (u == node and v in successors):
            weight = attr['weight']
            weights.append(sum(weight))
    #outgoing_weight = sum(sum(G11.edges[node, neighbor]['weight']) for neighbor in G11.successors(node))
    outgoing_weighted_degrees[node] = sum(weights)

for node in G11.nodes():
    predecessors = []
    for node_id in G11.predecessors(node):
        predecessors.append(node_id)
    weights = []
    for u, v, key, attr in G11.edges(keys=True, data=True):
        if (u in predecessors and v == node) or (u == node and v in predecessors):
            weight = attr['weight']
            weights.append(sum(weight))
    #incoming_weight = sum(sum(G11.edges[neighbor, node]['weight']) for neighbor in G11.predecessors(node))
    incoming_weighted_degrees[node] = sum(weights)

for node in G11.nodes():
    overall_weighted_degree = outgoing_weighted_degrees.get(node, 0) + incoming_weighted_degrees.get(node, 0)
    overall_weighted_degrees[node] = overall_weighted_degree

sorted_outgoing_weighted_degrees = dict(sorted(outgoing_weighted_degrees.items(), key=lambda item: item[1], reverse=True))
sorted_incoming_weighted_degrees = dict(sorted(incoming_weighted_degrees.items(), key=lambda item: item[1], reverse=True))
sorted_overall_weighted_degrees = dict(sorted(overall_weighted_degrees.items(), key=lambda item: item[1], reverse=True))

print("Outgoing Weighted Degrees:", sorted_outgoing_weighted_degrees)
print("Incoming Weighted Degrees:", sorted_incoming_weighted_degrees)
print("Overall Weighted Degrees:", sorted_overall_weighted_degrees)


# In[92]:


from collections import Counter


source_place_counter = Counter()
target_place_counter = Counter()
overall_place_counter = Counter()


for publisher_A, publisher_B, data in G.edges(data=True):
    if isinstance(data['source_place'], list):
        source_places = data['source_place']
    else:
        source_places = [data['source_place']]
    if isinstance(data['target_place'], list):
        target_places = data['target_place']
    else:
        target_places = [data['target_place']]

    
    for i, place_list in enumerate(source_places):
        if isinstance(place_list, str):
            source_places[i] = [place_list]
    
    for i, place_list in enumerate(target_places):
        if isinstance(place_list, str):
            #place_list = [place_list]
            target_places[i] = [place_list]
            
    for place_list in source_places:
        for place in place_list:
            source_place_counter[place] += 1
            
    for place_list in target_places:
        for place in place_list:
            target_place_counter[place] += 1
    
    for place_list in source_places + target_places:
        #if isinstance(place_list, str):
            #place_list = [place_list]  # Convert string to list
        for place in place_list:
            overall_place_counter[place] += 1


sorted_source_places = source_place_counter.most_common()
sorted_target_places = target_place_counter.most_common()
sorted_overall_places = overall_place_counter.most_common()


print("Sorted Source Places:", sorted_source_places)
print("Sorted Target Places:", sorted_target_places)
print("Sorted Overall Places:", sorted_overall_places)


# In[21]:


publisher_C = "James Duffy"
publisher_D = "Benziger Brothers"

for publisher_C, publisher_D, data in G.edges(data=True):
    weight = data['weight']
    print(weight)


# In[93]:


source_place_counter_weighted = Counter()
target_place_counter_weighted = Counter()
overall_place_counter_weighted = Counter()

for publisher_A, publisher_B, data in G.edges(data=True):
    # Get the weight of the current edge
    weight = data['weight']
    
    if isinstance(data['source_place'], list):
        source_places = data['source_place']
    else:
        source_places = [data['source_place']]
    if isinstance(data['target_place'], list):
        target_places = data['target_place']
    else:
        target_places = [data['target_place']]

    for i, place_list in enumerate(source_places):
        if isinstance(place_list, str):
            source_places[i] = [place_list]

    for i, place_list in enumerate(target_places):
        if isinstance(place_list, str):
            target_places[i] = [place_list]

    for place_list in source_places:
        for place in place_list:
            source_place_counter_weighted[place] += weight[0]

    for place_list in target_places:
        for place in place_list:
            target_place_counter_weighted[place] += weight[0]

    for place_list in source_places + target_places:
        for place in place_list:
            overall_place_counter_weighted[place] += weight[0]

sorted_source_places_weighted = source_place_counter_weighted.most_common()
sorted_target_places_weighted = target_place_counter_weighted.most_common()
sorted_overall_places_weighted = overall_place_counter_weighted.most_common()

print("Sorted Source Places (Weighted):", sorted_source_places_weighted)
print("Sorted Target Places (Weighted):", sorted_target_places_weighted)
print("Sorted Overall Places (Weighted):", sorted_overall_places_weighted)


# In[94]:


# Initialize counters for unweighted and weighted titles
unweighted_title_counter = Counter()
weighted_title_counter = Counter()

# Initialize dictionaries to store concatenated source and target places for each title
title_source_places = {}
title_target_places = {}

# Iterate over edges in the graph
for publisher_A, publisher_B, data in G.edges(data=True):
    title = data['title']
    weight = data['weight']
    
    # Increment the unweighted and weighted title counters
    unweighted_title_counter[title] += 1
    weighted_title_counter[title] += weight[0]
    
    if isinstance(data['source_place'], list):
        source_places = data['source_place']
    else:
        source_places = [data['source_place']]
    if isinstance(data['target_place'], list):
        target_places = data['target_place']
    else:
        target_places = [data['target_place']]

    for i, place_list in enumerate(source_places):
        if isinstance(place_list, str):
            source_places[i] = [place_list]

    for i, place_list in enumerate(target_places):
        if isinstance(place_list, str):
            target_places[i] = [place_list]
    
    if title not in title_source_places:
                title_source_places[title] = [] 
            
    for place_list in source_places:
        for place in place_list:
            title_source_places[title].append(place)
    
    if title not in title_target_places:
                title_target_places[title] = [] 
            
    for place_list in target_places:
        for place in place_list:          
            title_target_places[title].append(place)
            
# Sort titles based on unweighted occurrences
sorted_unweighted_titles = unweighted_title_counter.most_common()

# Sort titles based on weighted occurrences
sorted_weighted_titles = weighted_title_counter.most_common()

for elem in sorted_unweighted_titles:
    #source_places = ', '.join(title_source_places[title])
    #target_places = ', '.join(title_target_places[title])
    #print(f"Title: {title} | Source Places: {source_places} | Target Places: {target_places}")
    print(f"Title and number of connections between publishers it occasioned: {elem} | Source Places: {set(title_source_places[elem[0]])} | Target Places: {set(title_target_places[elem[0]])}")

for ele in sorted_weighted_titles:
    #source_places = ', '.join(title_source_places[title])
    #target_places = ', '.join(title_target_places[title])
    print(f"Title and weighted number of connections between publishers it occasioned: {ele} | Source Places: {set(title_source_places[ele[0]])} | Target Places: {set(title_target_places[ele[0]])}")


# In[95]:


G_combined = nx.DiGraph()

for u, v, key, attr in G.edges(keys=True, data=True):
    if G_combined.has_edge(u, v):
        G_combined[u][v]['weight'] += sum(data['weight'])
    else:
        G_combined.add_edge(u, v, weight=sum(data['weight']))

for node, data in G.nodes(data=True):
    if not G_combined.has_node(node):
        G_combined.add_node(node, magnitude=G.nodes[node].get('magnitude', 1))


# In[96]:


len(G_combined.nodes())


# In[28]:


len(G_combined.edges())


# In[98]:


closeness_centralities = nx.closeness_centrality(G_combined)

betweenness_centralities = nx.betweenness_centrality(G_combined)

eigenvector_centralities = nx.eigenvector_centrality(G_combined)


# In[105]:


eigenvector_centralities = nx.eigenvector_centrality(G_combined, max_iter = 600)


# In[35]:


import operator


# In[106]:


sorted_closeness = sorted(closeness_centralities.items(), key=operator.itemgetter(1), reverse=True)
sorted_betweenness = sorted(betweenness_centralities.items(), key=operator.itemgetter(1), reverse=True)
sorted_eigenvector = sorted(eigenvector_centralities.items(), key=operator.itemgetter(1), reverse=True)


# In[107]:


sorted_closeness


# In[108]:


sorted_betweenness


# In[109]:


sorted_eigenvector


# In[112]:


weighted_closeness = nx.closeness_centrality(G_combined, distance='weight')
weighted_betweenness = nx.betweenness_centrality(G_combined, weight='weight')
weighted_eigenvector = nx.eigenvector_centrality(G_combined, max_iter = 200, weight='weight')

sorted_weighted_closeness = sorted(weighted_closeness.items(), key=lambda x: x[1], reverse=True)
sorted_weighted_betweenness = sorted(weighted_betweenness.items(), key=lambda x: x[1], reverse=True)
sorted_weighted_eigenvector = sorted(weighted_eigenvector.items(), key=lambda x: x[1], reverse=True)


# In[113]:


sorted_weighted_closeness


# In[114]:


sorted_weighted_betweenness


# In[115]:


sorted_weighted_eigenvector


# In[119]:


import matplotlib.pyplot as plt


# In[46]:


main_component = max(nx.connected_components(G_combined), key=len)
G_main = G_combined.subgraph(main_component)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G_main)
edge_widths = [data['weight'] for _, _, data in G_main.edges(data=True)]
nx.draw(G_main, pos, with_labels=True, width=edge_widths, edge_color='gray')

# Save the plot as a standalone file (e.g., PNG format)
plt.savefig('Pietra_Margento_19th_Cent_Liguori_EN_main_component_graph_updated.png')

# Display the plot
plt.show()


# In[127]:


#import networkx as nx
#import matplotlib.pyplot as plt

G_combined = nx.DiGraph()

for u, v, data in G.edges(data=True):
    if G_combined.has_edge(u, v):
        # If the edge exists, update the weight attribute by adding the current edge's weight
        G_combined[u][v]['weight'] += data['weight']
    else:
        G_combined.add_edge(u, v, weight=data['weight'])

    if not G_combined.has_node(u):
        G_combined.add_node(u, magnitude=G.nodes[u].get('magnitude', 1))  # Default magnitude to 1 if not present
    else:
        G_combined.nodes[u]['magnitude'] = G.nodes[u].get('magnitude', 1)

    if not G_combined.has_node(v):
        G_combined.add_node(v, magnitude=G.nodes[v].get('magnitude', 1))  # Default magnitude to 1 if not present
    else:
        G_combined.nodes[v]['magnitude'] = G.nodes[v].get('magnitude', 1)

for u, v, data in G_combined.edges(data=True):
    data['weight'] = sum(data['weight'])

for node, data in G.nodes(data=True):
    if not G_combined.has_node(node):
        G_combined.add_node(node, magnitude=G.nodes[node].get('magnitude', 1))
        
node_sizes = [G_combined.nodes[node].get('magnitude', 1) for node in G_combined.nodes()]

# Scale node sizes for better visualization (adjust scaling factor as needed)
node_sizes = [size * 36 for size in node_sizes]

layout = nx.spring_layout(G_combined)

edge_widths = [data['weight'] for _, _, data in G_combined.edges(data=True)]

# Plot the graph
plt.figure(figsize=(12, 8))
nx.draw(G_combined, pos=layout, node_size=node_sizes, with_labels=True,
        alpha=1, node_color='skyblue', width=edge_widths, edge_color='gray', arrows=True)

plt.title("Pietra 19th Cent Liguori English Translation Network with Node Magnitudes")

plt.savefig('Pietra_Margento_19th_Cent_Liguori_EN_w_Node_Magnitudes.png')

plt.show()


# In[125]:


main_component = max(nx.weakly_connected_components(G_combined), key=len)
G_main = G_combined.subgraph(main_component)


# In[130]:


node_sizes = [G_main.nodes[node]['magnitude'] for node in G_main.nodes()]

# Scale node sizes for better visualization (adjust scaling factor as needed)
node_sizes = [size * 36 for size in node_sizes]

layout = nx.spring_layout(G_main)

edge_widths = [data['weight'] for _, _, data in G_main.edges(data=True)]

# Plot the graph
plt.figure(figsize=(14, 10))
nx.draw(G_main, pos=layout, node_size=node_sizes, with_labels=True,
        alpha=1, node_color='skyblue', width=edge_widths, edge_color='gray', arrows=True)

plt.title("Pietra 19th Cent Liguori English Translation Main Weakly Connected Component")

plt.savefig('Pietra_Margento_19th_Cent_Liguori_EN_Main_Weakly_Connected_Component.png')

plt.show()


# In[ ]:




