#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import networkx as nx
import itertools
import matplotlib.pyplot as plt


# In[2]:


excel_file = '19th_century_translations.xlsx'
df = pd.read_excel(excel_file)


# In[69]:


#import pandas as pd
#import networkx as nx

G_1861_80 = nx.MultiDiGraph()

# Function to extract and sort years chronologically
def extract_years(row, cutoff_year0, cutoff_year1):
    year = row['Year']
    earlier = row['Earlier']
    subsequent = row['Subsequent Listing']
    
    # Extract and filter years based on the cutoff year
    all_years = []
    #if isinstance(year, (int, float)) and year > cutoff_year0 and year < cutoff_year1:
    if year not in ['[s.d.]', '185?', '18--', '????'] and int(year) > cutoff_year0 and int(year) < cutoff_year1:
        all_years.append(int(year))
    
    if isinstance(earlier, str) and earlier != 'nan':
        earlier_years = [int(x.split()[0]) for x in earlier.split(',') if int(x.split()[0]) < cutoff_year1 and int(x.split()[0]) > cutoff_year0]
        all_years.extend(earlier_years)
    
    if isinstance(subsequent, str) and subsequent != 'nan':
        subsequent_years = [int(x.split()[0]) for x in subsequent.split(',') if int(x.split()[0]) < cutoff_year1 and int(x.split()[0]) > cutoff_year0]
        all_years.extend(subsequent_years)
    
    all_years = sorted(all_years)
    return all_years

filtered_df0 = df.iloc[0:179]

def is_valid_year(x):
    try:
        year_int = int(x)
        return year_int > 1860 and year_int < 1881
    except ValueError:
        return False

# Filter rows based on 'Year' column
filtered_df = filtered_df0[filtered_df0['Year'].apply(lambda x: x not in ['[s.d.]', '185?', '18--', '????'] and is_valid_year(x))]

unique_publishers = set()

for publishers in filtered_df['Publisher']:
    if isinstance(publishers, str):
        unique_publishers.update([publisher.strip() for publisher in publishers.split(';')])
    else:
        unique_publishers.update([publishers])

for publisher in unique_publishers:
    G_1861_80.add_node(publisher, publications=[])

for index, row in filtered_df.iterrows():
    title = row['Title']
    years = extract_years(row, 1860, 1881)
    editions = sum(1 for year in years)
    publishers = [publisher.strip() for publisher in row['Publisher'].split('; ')] if isinstance(row['Publisher'], str) else [row['Publisher']]
    translator = row['Translator']
    places = [place.strip() for place in row['Place of Publication'].split('; ')] if isinstance(row['Place of Publication'], str) else [str(row['Place of Publication']).strip()]

    for publisher in unique_publishers:
        if publisher in publishers:
            G_1861_80.nodes[publisher]['publications'].append({
                'title': title,
                'years': years,
                'editions': editions,
                'publisher': publishers,
                'translator': translator,
                'place': places
            })

for publisher in G_1861_80.nodes():
    magnitude = sum(pub['editions'] for pub in G_1861_80.nodes[publisher]['publications'])
    G_1861_80.nodes[publisher]['magnitude'] = magnitude


# In[68]:


filtered_df0


# In[70]:


def add_edge_with_attributes(G, publisher_A, publisher_B, title, source_years, target_years,
                             source_translator, target_translator, source_num_editions,
                             target_num_editions, source_place, target_place, weight, relation, key):
    G.add_edge(publisher_A, publisher_B, title=title, source_years=source_years, target_years=target_years,
               source_translator=source_translator, target_translator=target_translator,
               source_num_editions=source_num_editions, target_num_editions=target_num_editions,
               source_place=source_place, target_place=target_place, weight=weight, relation=relation, key=key)


# In[71]:


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
for publisher_A, publisher_B in itertools.combinations(G_1861_80.nodes(), 2):
    publications_A = G_1861_80.nodes[publisher_A]["publications"]
    publications_B = G_1861_80.nodes[publisher_B]["publications"]
    
    # Iterate over publications of each publisher pair
    for publication_A in publications_A:
        for publication_B in publications_B:
            # Add edges based on different relations
            add_edges_between_publishers(G_1861_80, publisher_A, publisher_B, publication_A, publication_B)


# In[72]:


sorted_nodes = sorted(G_1861_80.nodes(data=True), key=lambda x: x[1].get('magnitude', 0), reverse=True)

# Print nodes in descending order of magnitude
for node, attrs in sorted_nodes:
    print(f"{node}, Magnitude: {attrs.get('magnitude', 0)}")


# In[73]:


G_1861_80.nodes['James Duffy']['publications']


# In[65]:


for node, data in G_1861_80.nodes(data=True):
    for publication in data['publications']:
        print(f"{node}", publication['place'])


# In[75]:


len(G_1861_80.edges())


# In[74]:


out_degrees_80 = dict(G_1861_80.out_degree())
in_degrees_80 = dict(G_1861_80.in_degree())
overall_degrees_80 = {node: out_degrees_80.get(node, 0) + in_degrees_80.get(node, 0) for node in G_1861_80.nodes()}

nodes_by_out_degree_80 = sorted(out_degrees_80, key=out_degrees_80.get, reverse=True)
nodes_by_in_degree_80 = sorted(in_degrees_80, key=in_degrees_80.get, reverse=True)
nodes_by_overall_degree_80 = sorted(overall_degrees_80, key=overall_degrees_80.get, reverse=True)

print("Nodes ordered by outgoing degree:")
for node in nodes_by_out_degree_80:
    print(node, out_degrees_80[node])

print("\nNodes ordered by incoming degree:")
for node in nodes_by_in_degree_80:
    print(node, in_degrees_80[node])

print("\nNodes ordered by overall degree:")
for node in nodes_by_overall_degree_80:
    print(node, overall_degrees_80[node])


# In[80]:


for u, v, key, attr in G_1861_80.edges(keys=True, data=True):
    print(attr['weight'])


# In[81]:


outgoing_weighted_degrees = {}
incoming_weighted_degrees = {}
overall_weighted_degrees = {}

for node in G_1861_80.nodes():
    successors = []
    for node_id in G_1861_80.successors(node):
        successors.append(node_id)
    weights = []
    for u, v, key, attr in G_1861_80.edges(keys=True, data=True):
        if (u in successors and v == node) or (u == node and v in successors):
            weight = attr['weight']
            weights.append(weight)
    #outgoing_weight = sum(sum(G11.edges[node, neighbor]['weight']) for neighbor in G11.successors(node))
    outgoing_weighted_degrees[node] = sum(weights)

for node in G_1861_80.nodes():
    predecessors = []
    for node_id in G_1861_80.predecessors(node):
        predecessors.append(node_id)
    weights = []
    for u, v, key, attr in G_1861_80.edges(keys=True, data=True):
        if (u in predecessors and v == node) or (u == node and v in predecessors):
            weight = attr['weight']
            weights.append(weight)
    #incoming_weight = sum(sum(G11.edges[neighbor, node]['weight']) for neighbor in G11.predecessors(node))
    incoming_weighted_degrees[node] = sum(weights)

for node in G_1861_80.nodes():
    overall_weighted_degree = outgoing_weighted_degrees.get(node, 0) + incoming_weighted_degrees.get(node, 0)
    overall_weighted_degrees[node] = overall_weighted_degree

sorted_outgoing_weighted_degrees = dict(sorted(outgoing_weighted_degrees.items(), key=lambda item: item[1], reverse=True))
sorted_incoming_weighted_degrees = dict(sorted(incoming_weighted_degrees.items(), key=lambda item: item[1], reverse=True))
sorted_overall_weighted_degrees = dict(sorted(overall_weighted_degrees.items(), key=lambda item: item[1], reverse=True))

print("Outgoing Weighted Degrees:", sorted_outgoing_weighted_degrees)
print("Incoming Weighted Degrees:", sorted_incoming_weighted_degrees)
print("Overall Weighted Degrees:", sorted_overall_weighted_degrees)


# In[82]:


from collections import Counter


source_place_counter = Counter()
target_place_counter = Counter()
overall_place_counter = Counter()


for publisher_A, publisher_B, data in G_1861_80.edges(data=True):
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


sorted_source_places_80 = source_place_counter.most_common()
sorted_target_places_80 = target_place_counter.most_common()
sorted_overall_places_80 = overall_place_counter.most_common()


print("Sorted Source Places:", sorted_source_places_80)
print("Sorted Target Places:", sorted_target_places_80)
print("Sorted Overall Places:", sorted_overall_places_80)


# In[85]:


source_place_counter_weighted = Counter()
target_place_counter_weighted = Counter()
overall_place_counter_weighted = Counter()

for publisher_A, publisher_B, data in  G_1861_80.edges(data=True):
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
            source_place_counter_weighted[place] += weight

    for place_list in target_places:
        for place in place_list:
            target_place_counter_weighted[place] += weight

    for place_list in source_places + target_places:
        for place in place_list:
            overall_place_counter_weighted[place] += weight

sorted_source_places_weighted_80 = source_place_counter_weighted.most_common()
sorted_target_places_weighted_80 = target_place_counter_weighted.most_common()
sorted_overall_places_weighted_80 = overall_place_counter_weighted.most_common()

print("Sorted Source Places (Weighted):", sorted_source_places_weighted_80)
print("Sorted Target Places (Weighted):", sorted_target_places_weighted_80)
print("Sorted Overall Places (Weighted):", sorted_overall_places_weighted_80)


# In[87]:


#from collections import Counter

# Initialize counters for unweighted and weighted titles
unweighted_title_counter = Counter()
weighted_title_counter = Counter()

# Initialize dictionaries to store concatenated source and target places for each title
title_source_places = {}
title_target_places = {}

# Iterate over edges in the graph
for publisher_A, publisher_B, data in G_1861_80.edges(data=True):
    title = data['title']
    weight = data['weight']
    
    # Increment the unweighted and weighted title counters
    unweighted_title_counter[title] += 1
    weighted_title_counter[title] += weight
    
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
            
    
sorted_unweighted_titles_80 = unweighted_title_counter.most_common()

# Sort titles based on weighted occurrences
sorted_weighted_titles_80 = weighted_title_counter.most_common()

for elem in sorted_unweighted_titles_80:
    print(f"Title and number of connections between publishers it occasioned: {elem} | Source Places: {set(title_source_places[elem[0]])} | Target Places: {set(title_target_places[elem[0]])}")

for ele in sorted_weighted_titles_80:

    print(f"Title and weighted number of connections between publishers it occasioned: {ele} | Source Places: {set(title_source_places[ele[0]])} | Target Places: {set(title_target_places[ele[0]])}")


# In[88]:


for u, v, data in G_1861_80.edges(data = True):
    #if (('Dublin' in data['target_place']) or (data['target_place'] == 'Dublin')):
        print(u, "and", v, "for", data['title'], "from", data['source_place'], "to", data['target_place'], "as", data['relation'])


# In[32]:


G_combined_80 = nx.DiGraph()

for u, v, data in G_1861_80.edges(data=True):
    if G_combined_80.has_edge(u, v):
        # If the edge exists, update the weight attribute by adding the current edge's weight
        G_combined_80[u][v]['weight'] += data['weight']
    else:
        G_combined_80.add_edge(u, v, weight=data['weight'])

    if not G_combined_80.has_node(u):
        G_combined_80.add_node(u, magnitude=G_1861_80.nodes[u].get('magnitude', 1))  # Default magnitude to 1 if not present
    else:
        G_combined_80.nodes[u]['magnitude'] = G_1861_80.nodes[u].get('magnitude', 1)

    if not G_combined_80.has_node(v):
        G_combined_80.add_node(v, magnitude=G_1861_80.nodes[v].get('magnitude', 1))  # Default magnitude to 1 if not present
    else:
        G_combined_80.nodes[v]['magnitude'] = G_1861_80.nodes[v].get('magnitude', 1)

for u, v, data in G_combined_80.edges(data=True):
    data['weight'] = sum(data['weight'])

for node, data in G_1861_80.nodes(data=True):
    if not G_combined_80.has_node(node):
        G_combined_80.add_node(node, magnitude=G_1861_80.nodes[node].get('magnitude', 1))
    
node_sizes = [G_combined_80.nodes[node].get('magnitude', 1) for node in G_combined_80.nodes()]


# In[33]:


closeness_centralities_2 = nx.closeness_centrality(G_combined_80)

betweenness_centralities_2 = nx.betweenness_centrality(G_combined_80)

eigenvector_centralities_2 = nx.eigenvector_centrality(G_combined_80)


# In[34]:


import operator

sorted_closeness_2 = sorted(closeness_centralities_2.items(), key=operator.itemgetter(1), reverse=True)
sorted_betweenness_2 = sorted(betweenness_centralities_2.items(), key=operator.itemgetter(1), reverse=True)
sorted_eigenvector_2 = sorted(eigenvector_centralities_2.items(), key=operator.itemgetter(1), reverse=True)


# In[35]:


sorted_closeness_2


# In[36]:


sorted_betweenness_2


# In[37]:


sorted_eigenvector_2


# In[15]:


for u, v, data in G_combined_80.edges(data=True):
    print(data['weight'])


# In[38]:


weighted_closeness_2 = nx.closeness_centrality(G_combined_80, distance='weight')
weighted_betweenness_2 = nx.betweenness_centrality(G_combined_80, weight='weight')
weighted_eigenvector_2 = nx.eigenvector_centrality(G_combined_80, weight='weight')

sorted_weighted_closeness_2 = sorted(weighted_closeness_2.items(), key=lambda x: x[1], reverse=True)
sorted_weighted_betweenness_2 = sorted(weighted_betweenness_2.items(), key=lambda x: x[1], reverse=True)
sorted_weighted_eigenvector_2 = sorted(weighted_eigenvector_2.items(), key=lambda x: x[1], reverse=True)


# In[18]:


sorted_weighted_closeness_2


# In[122]:


sorted_weighted_closeness_2 == sorted_closeness_2


# In[123]:


sorted_weighted_betweenness_2 == sorted_betweenness_2


# In[124]:


sorted_weighted_eigenvector_2 == sorted_eigenvector_2


# In[128]:


for u, v, data in G_1861_80.edges(data=True):
    print(data['weight'])


# In[130]:


node_sizes = [G_1861_80.nodes[node].get('magnitude', 1) for node in G_combined_80.nodes()]

# Scale node sizes for better visualization (adjust scaling factor as needed)
node_sizes = [size * 36 for size in node_sizes]

layout = nx.spring_layout(G_combined_80)

edge_widths = [data['weight'] for _, _, data in G_1861_80.edges(data=True)]

# Plot the graph
plt.figure(figsize=(12, 8))
nx.draw(G_1861_80, pos=layout, node_size=node_sizes, with_labels=True,
        alpha=1, node_color='skyblue', width=edge_widths, edge_color='gray', arrows=True)

plt.title("Pietra 19th Cent Liguori English Translation Network 1861-1880")

plt.savefig('Pietra_Margento_19th_Cent_Liguori_EN_1861-1880.png')

plt.show()


# In[131]:


l1 = list(nx.weakly_connected_components(G_1861_80))


# In[132]:


len(l1)


# In[133]:


Go = G_1861_80.subgraph(l1[0])

node_sizes = [Go.nodes[node].get('magnitude', 1) for node in Go.nodes()]

# Scale node sizes for better visualization (adjust scaling factor as needed)
node_sizes = [size * 36 for size in node_sizes]

layout = nx.spring_layout(Go)

edge_widths = [data['weight'] for _, _, data in Go.edges(data=True)]

# Plot the graph
plt.figure(figsize=(12, 8))
nx.draw(Go, pos=layout, node_size=node_sizes, with_labels=True,
        alpha=1, node_color='skyblue', width=edge_widths, edge_color='gray', arrows=True)

plt.title("19th Cent Liguori 1861-1880 First Component")

plt.savefig('19th_Cent_Liguori_EN_1861-1880_First_Component.png')

plt.show()


# In[24]:


Go = G_combined_80.subgraph(l1[1])

node_sizes = [Go.nodes[node].get('magnitude', 1) for node in Go.nodes()]

# Scale node sizes for better visualization (adjust scaling factor as needed)
node_sizes = [size * 36 for size in node_sizes]

layout = nx.spring_layout(Go)

edge_widths = [data['weight'] for _, _, data in Go.edges(data=True)]

# Plot the graph
plt.figure(figsize=(12, 8))
nx.draw(Go, pos=layout, node_size=node_sizes, with_labels=True,
        alpha=1, node_color='skyblue', width=edge_widths, edge_color='gray', arrows=True)

plt.title("19th Cent Liguori 1861-1880 Second Component")

plt.savefig('19th_Cent_Liguori_EN_1861-1880_Second_Component.png')

plt.show()


# In[134]:


Go = G_1861_80.subgraph(l1[2])

node_sizes = [Go.nodes[node].get('magnitude', 1) for node in Go.nodes()]

# Scale node sizes for better visualization (adjust scaling factor as needed)
node_sizes = [size * 36 for size in node_sizes]

layout = nx.spring_layout(Go)

edge_widths = [data['weight'] for _, _, data in Go.edges(data=True)]

# Plot the graph
plt.figure(figsize=(12, 8))
nx.draw(Go, pos=layout, node_size=node_sizes, with_labels=True,
        alpha=1, node_color='skyblue', width=edge_widths, edge_color='gray', arrows=True)

plt.title("19th Cent Liguori 1861-1880 Second Component")

plt.savefig('19th_Cent_Liguori_EN_1861-1880_Second_Component.png')

plt.show()


# In[ ]:





# In[93]:


#import pandas as pd
#import networkx as nx

G_after_1880 = nx.MultiDiGraph()

# Function to extract and sort years chronologically
def extract_years(row, cutoff_year):
    year = row['Year']
    earlier = row['Earlier']
    subsequent = row['Subsequent Listing']
    
    # Extract and filter years based on the cutoff year
    all_years = []
    #if isinstance(year, (int, float)) and year > cutoff_year0 and year < cutoff_year1:
    if year not in ['[s.d.]', '185?', '18--', '????'] and int(year) > cutoff_year:
        all_years.append(int(year))
    
    if isinstance(earlier, str) and earlier != 'nan':
        earlier_years = [int(x.split()[0]) for x in earlier.split(',') if x.split()[0] not in ['6th', '7th'] and int(x.split()[0]) > cutoff_year]
        all_years.extend(earlier_years)
    
    if isinstance(subsequent, str) and subsequent != 'nan':
        subsequent_years = [int(x.split()[0]) for x in subsequent.split(',') if int(x.split()[0]) > cutoff_year]
        all_years.extend(subsequent_years)
    
    all_years = sorted(all_years)
    return all_years

filtered_df0 = df.iloc[0:179]

def is_valid_year(x):
    try:
        year_int = int(x)
        return year_int > 1880
    except ValueError:
        return False

# Filter rows based on 'Year' column
filtered_df = filtered_df0[filtered_df0['Year'].apply(lambda x: x not in ['[s.d.]', '185?', '18--', '????'] and is_valid_year(x))]

unique_publishers = set()
for publishers in filtered_df['Publisher']:
    if isinstance(publishers, str):
        unique_publishers.update([publisher.strip() for publisher in publishers.split(';')])
    else:
        unique_publishers.update([publishers])

for publisher in unique_publishers:
    G_after_1880.add_node(publisher, publications=[])

for index, row in filtered_df.iterrows():
    title = row['Title']
    years = extract_years(row, 1880)
    editions = sum(1 for year in years) if years != [] else 0  # Calculate number of editions based on years before 1851
    publishers = [publisher.strip() for publisher in row['Publisher'].split('; ')] if isinstance(row['Publisher'], str) else [row['Publisher']]
    translator = row['Translator']
    places = [place.strip() for place in row['Place of Publication'].split('; ')] if isinstance(row['Place of Publication'], str) else [str(row['Place of Publication']).strip()]

    for publisher in unique_publishers:
        if publisher in publishers:
            G_after_1880.nodes[publisher]['publications'].append({
                'title': title,
                'years': years,
                'editions': [editions],
                'publisher': publishers,
                'translator': translator,
                'place': places
            })

for publisher in G_after_1880.nodes():
    if 'publications' in G_after_1880.nodes[publisher]:  # Check if 'publications' key exists
        magnitude = sum(pub['editions'][0] for pub in G_after_1880.nodes[publisher]['publications'])
        G_after_1880.nodes[publisher]['magnitude'] = magnitude


# In[95]:


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
for publisher_A, publisher_B in itertools.combinations(G_after_1880.nodes(), 2):
    publications_A = G_after_1880.nodes[publisher_A]["publications"]
    publications_B = G_after_1880.nodes[publisher_B]["publications"]
    
    # Iterate over publications of each publisher pair
    for publication_A in publications_A:
        for publication_B in publications_B:
            # Add edges based on different relations
            add_edges_between_publishers(G_after_1880, publisher_A, publisher_B, publication_A, publication_B)


# In[96]:


sorted_nodes = sorted(G_after_1880.nodes(data=True), key=lambda x: x[1].get('magnitude', 0), reverse=True)

# Print nodes in descending order of magnitude
for node, attrs in sorted_nodes:
    print(f"{node}, Magnitude: {attrs.get('magnitude', 0)}")


# In[98]:


len(G_after_1880.edges())


# In[97]:


out_degrees_00 = dict(G_after_1880.out_degree())
in_degrees_00 = dict(G_after_1880.in_degree())
overall_degrees_00 = {node: out_degrees_00.get(node, 0) + in_degrees_00.get(node, 0) for node in G_after_1880.nodes()}

nodes_by_out_degree_00 = sorted(out_degrees_00, key=out_degrees_00.get, reverse=True)
nodes_by_in_degree_00 = sorted(in_degrees_00, key=in_degrees_00.get, reverse=True)
nodes_by_overall_degree_00 = sorted(overall_degrees_00, key=overall_degrees_00.get, reverse=True)

print("Nodes ordered by outgoing degree:")
for node in nodes_by_out_degree_00:
    print(node, out_degrees_00[node])

print("\nNodes ordered by incoming degree:")
for node in nodes_by_in_degree_00:
    print(node, in_degrees_00[node])

print("\nNodes ordered by overall degree:")
for node in nodes_by_overall_degree_00:
    print(node, overall_degrees_00[node])


# In[99]:


outgoing_weighted_degrees = {}
incoming_weighted_degrees = {}
overall_weighted_degrees = {}

for node in G_after_1880.nodes():
    successors = []
    for node_id in G_after_1880.successors(node):
        successors.append(node_id)
    weights = []
    for u, v, key, attr in G_after_1880.edges(keys=True, data=True):
        if (u in successors and v == node) or (u == node and v in successors):
            weight = attr['weight']
            weights.append(sum(weight))
    #outgoing_weight = sum(sum(G11.edges[node, neighbor]['weight']) for neighbor in G11.successors(node))
    outgoing_weighted_degrees[node] = sum(weights)

for node in G_after_1880.nodes():
    predecessors = []
    for node_id in G_after_1880.predecessors(node):
        predecessors.append(node_id)
    weights = []
    for u, v, key, attr in G_after_1880.edges(keys=True, data=True):
        if (u in predecessors and v == node) or (u == node and v in predecessors):
            weight = attr['weight']
            weights.append(sum(weight))
    #incoming_weight = sum(sum(G11.edges[neighbor, node]['weight']) for neighbor in G11.predecessors(node))
    incoming_weighted_degrees[node] = sum(weights)

for node in G_after_1880.nodes():
    overall_weighted_degree = outgoing_weighted_degrees.get(node, 0) + incoming_weighted_degrees.get(node, 0)
    overall_weighted_degrees[node] = overall_weighted_degree

sorted_outgoing_weighted_degrees = dict(sorted(outgoing_weighted_degrees.items(), key=lambda item: item[1], reverse=True))
sorted_incoming_weighted_degrees = dict(sorted(incoming_weighted_degrees.items(), key=lambda item: item[1], reverse=True))
sorted_overall_weighted_degrees = dict(sorted(overall_weighted_degrees.items(), key=lambda item: item[1], reverse=True))

print("Outgoing Weighted Degrees:", sorted_outgoing_weighted_degrees)
print("Incoming Weighted Degrees:", sorted_incoming_weighted_degrees)
print("Overall Weighted Degrees:", sorted_overall_weighted_degrees)


# In[100]:


from collections import Counter


source_place_counter = Counter()
target_place_counter = Counter()
overall_place_counter = Counter()


for publisher_A, publisher_B, data in G_after_1880.edges(data=True):
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


sorted_source_places_0 = source_place_counter.most_common()
sorted_target_places_0 = target_place_counter.most_common()
sorted_overall_places_0 = overall_place_counter.most_common()


print("Sorted Source Places:", sorted_source_places_0)
print("Sorted Target Places:", sorted_target_places_0)
print("Sorted Overall Places:", sorted_overall_places_0)


# In[103]:


source_place_counter_weighted = Counter()
target_place_counter_weighted = Counter()
overall_place_counter_weighted = Counter()

for publisher_A, publisher_B, data in  G_after_1880.edges(data=True):
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

sorted_source_places_weighted_80 = source_place_counter_weighted.most_common()
sorted_target_places_weighted_80 = target_place_counter_weighted.most_common()
sorted_overall_places_weighted_80 = overall_place_counter_weighted.most_common()

print("Sorted Source Places (Weighted):", sorted_source_places_weighted_80)
print("Sorted Target Places (Weighted):", sorted_target_places_weighted_80)
print("Sorted Overall Places (Weighted):", sorted_overall_places_weighted_80)


# In[104]:


#from collections import Counter

# Initialize counters for unweighted and weighted titles
unweighted_title_counter = Counter()
weighted_title_counter = Counter()

# Initialize dictionaries to store concatenated source and target places for each title
title_source_places = {}
title_target_places = {}

# Iterate over edges in the graph
for publisher_A, publisher_B, data in G_after_1880.edges(data=True):
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
            
    
sorted_unweighted_titles_0 = unweighted_title_counter.most_common()

# Sort titles based on weighted occurrences
sorted_weighted_titles_0 = weighted_title_counter.most_common()

for elem in sorted_unweighted_titles_0:
    print(f"Title and number of connections between publishers it occasioned: {elem} | Source Places: {set(title_source_places[elem[0]])} | Target Places: {set(title_target_places[elem[0]])}")

for ele in sorted_weighted_titles_0:

    print(f"Title and weighted number of connections between publishers it occasioned: {ele} | Source Places: {set(title_source_places[ele[0]])} | Target Places: {set(title_target_places[ele[0]])}")


# In[105]:


G_combined_after_1880 = nx.DiGraph()

for u, v, data in G_after_1880.edges(data=True):
    if G_combined_after_1880.has_edge(u, v):
        # If the edge exists, update the weight attribute by adding the current edge's weight
        G_combined_after_1880[u][v]['weight'] += data['weight']
    else:
        G_combined_after_1880.add_edge(u, v, weight=data['weight'])

    if not G_combined_after_1880.has_node(u):
        G_combined_after_18800.add_node(u, magnitude=G_after_1880.nodes[u].get('magnitude', 1))  # Default magnitude to 1 if not present
    else:
        G_combined_after_1880.nodes[u]['magnitude'] = G_after_1880.nodes[u].get('magnitude', 1)

    if not G_combined_after_1880.has_node(v):
        G_combined_after_1880.add_node(v, magnitude=G_after_1880.nodes[v].get('magnitude', 1))  # Default magnitude to 1 if not present
    else:
        G_combined_after_1880.nodes[v]['magnitude'] = G_after_1880.nodes[v].get('magnitude', 1)

for u, v, data in G_combined_after_1880.edges(data=True):
    data['weight'] = sum(data['weight'])
    
for node, data in G_after_1880.nodes(data=True):
    if not G_combined_after_1880.has_node(node):
        G_combined_after_1880.add_node(node, magnitude=G_after_1880.nodes[node].get('magnitude', 1))
    
node_sizes = [G_combined_after_1880.nodes[node].get('magnitude', 1) for node in G_combined_after_1880.nodes()]


# In[29]:


G_combined_after_1880.nodes()


# In[43]:


# NOW IT'S OKAY
G_combined_after_1880.nodes()


# In[106]:


len(G_combined_after_1880.edges())


# In[107]:


closeness_centralities_3 = nx.closeness_centrality(G_combined_after_1880)

betweenness_centralities_3 = nx.betweenness_centrality(G_combined_after_1880)

eigenvector_centralities_3 = nx.eigenvector_centrality(G_combined_after_1880)


# In[113]:


eigenvector_centralities_3 = nx.eigenvector_centrality(G_combined_after_1880, max_iter = 700)


# In[114]:


# import operator

sorted_closeness_3 = sorted(closeness_centralities_3.items(), key=operator.itemgetter(1), reverse=True)
sorted_betweenness_3 = sorted(betweenness_centralities_3.items(), key=operator.itemgetter(1), reverse=True)
sorted_eigenvector_3 = sorted(eigenvector_centralities_3.items(), key=operator.itemgetter(1), reverse=True)


# In[115]:


sorted_closeness_3


# In[116]:


sorted_betweenness_3


# In[117]:


sorted_eigenvector_3


# In[118]:


weighted_closeness_3 = nx.closeness_centrality(G_combined_after_1880, distance='weight')
weighted_betweenness_3 = nx.betweenness_centrality(G_combined_after_1880, weight='weight')
weighted_eigenvector_3 = nx.eigenvector_centrality(G_combined_after_1880, weight='weight')

sorted_weighted_closeness_3 = sorted(weighted_closeness_3.items(), key=lambda x: x[1], reverse=True)
sorted_weighted_betweenness_3 = sorted(weighted_betweenness_3.items(), key=lambda x: x[1], reverse=True)
sorted_weighted_eigenvector_3 = sorted(weighted_eigenvector_3.items(), key=lambda x: x[1], reverse=True)


# In[51]:


sorted_weighted_closeness_3 == sorted_closeness_3


# In[119]:


sorted_weighted_closeness_3


# In[53]:


sorted_weighted_betweenness_3 == sorted_betweenness_3


# In[120]:


sorted_weighted_betweenness_3


# In[55]:


sorted_weighted_eigenvector_3 == sorted_eigenvector_3


# In[121]:


sorted_weighted_eigenvector_3 


# In[136]:


edge_widths = []
node_sizes = []


# In[137]:


G_combined_after_1880 = nx.DiGraph()

for u, v, data in G_after_1880.edges(data=True):
    if G_combined_after_1880.has_edge(u, v):
        # If the edge exists, update the weight attribute by adding the current edge's weight
        G_combined_after_1880[u][v]['weight'] += data['weight']
    else:
        G_combined_after_1880.add_edge(u, v, weight=data['weight'])

    if not G_combined_after_1880.has_node(u):
        G_combined_after_18800.add_node(u, magnitude=G_after_1880.nodes[u].get('magnitude', 1))  # Default magnitude to 1 if not present
    else:
        G_combined_after_1880.nodes[u]['magnitude'] = G_after_1880.nodes[u].get('magnitude', 1)

    if not G_combined_after_1880.has_node(v):
        G_combined_after_1880.add_node(v, magnitude=G_after_1880.nodes[v].get('magnitude', 1))  # Default magnitude to 1 if not present
    else:
        G_combined_after_1880.nodes[v]['magnitude'] = G_after_1880.nodes[v].get('magnitude', 1)

for u, v, data in G_combined_after_1880.edges(data=True):
    data['weight'] = sum(data['weight'])

for node, data in G_after_1880.nodes(data=True):
    if not G_combined_after_1880.has_node(node):
        G_combined_after_1880.add_node(node, magnitude=G_after_1880.nodes[node].get('magnitude', 1))
    
node_sizes = [G_combined_after_1880.nodes[node].get('magnitude', 1) for node in G_combined_after_1880.nodes()]

# Scale node sizes for better visualization (adjust scaling factor as needed)
node_sizes = [size * 36 for size in node_sizes]

layout = nx.spring_layout(G_combined_after_1880)

edge_widths = [data['weight']/8 for _, _, data in G_combined_after_1880.edges(data=True)]

# Plot the graph
plt.figure(figsize=(12, 8))
nx.draw(G_combined_after_1880, pos=layout, node_size=node_sizes, with_labels=True,
        alpha=1, node_color='skyblue', width=edge_widths, edge_color='gray', arrows=True)

plt.title("Pietra 19th Cent Liguori English Translation Network 1881 - ca. 1900")

plt.savefig('Pietra_Margento_19th_Cent_Liguori_EN_after_1880.png')

plt.show()


# In[138]:


l2 = list(nx.weakly_connected_components(G_combined_after_1880))


# In[139]:


len(l2)


# In[141]:


edge_widths = []


# In[142]:


Go = G_combined_after_1880.subgraph(l2[0])

node_sizes = [Go.nodes[node].get('magnitude', 1) for node in Go.nodes()]

# Scale node sizes for better visualization (adjust scaling factor as needed)
node_sizes = [size * 36 for size in node_sizes]

layout = nx.spring_layout(Go)

edge_widths = [data['weight']/8 for _, _, data in Go.edges(data=True)]

# Plot the graph
plt.figure(figsize=(12, 8))
nx.draw(Go, pos=layout, node_size=node_sizes, with_labels=True,
        alpha=1, node_color='skyblue', width=edge_widths, edge_color='gray', arrows=True)

plt.title("19th Cent Liguori 1880 - ca. 1900 First Component")

plt.savefig('19th_Cent_Liguori_EN_after_1880_First_Component.png')

plt.show()


# In[143]:


Go = G_combined_after_1880.subgraph(l2[1])

node_sizes = [Go.nodes[node].get('magnitude', 1) for node in Go.nodes()]

# Scale node sizes for better visualization (adjust scaling factor as needed)
node_sizes = [size * 36 for size in node_sizes]

layout = nx.spring_layout(Go)

edge_widths = [data['weight'] for _, _, data in Go.edges(data=True)]

# Plot the graph
plt.figure(figsize=(12, 8))
nx.draw(Go, pos=layout, node_size=node_sizes, with_labels=True,
        alpha=1, node_color='skyblue', width=edge_widths, edge_color='gray', arrows=True)

plt.title("19th Cent Liguori 1880 - ca. 1900 Second Component")

plt.savefig('19th_Cent_Liguori_EN_after_1880_Second_Component.png')

plt.show()


# In[ ]:




