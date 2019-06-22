"""
https://leetcode.com/problems/the-skyline-problem/

A city's skyline is the outer contour of the silhouette formed by all the buildings in that city when viewed from a distance.
Now suppose you are given the locations and height of all the buildings as shown on a cityscape photo (Figure A),
write a program to output the skyline formed by these buildings collectively (Figure B).

The geometric information of each building is represented by a triplet of integers [Li, Ri, Hi],
where Li and Ri are the x coordinates of the left and right edge of the ith building, respectively, and Hi is its height.

It is guaranteed that 0 ≤ Li, Ri ≤ INT_MAX, 0 < Hi ≤ INT_MAX, and Ri - Li > 0.
You may assume all buildings are perfect rectangles grounded on an absolutely flat surface at height 0.

For instance, the dimensions of all buildings in Figure A are recorded as: [ [2 9 10], [3 7 15], [5 12 12], [15 20 10], [19 24 8] ] .

The output is a list of "key points" (red dots in Figure B) in the format of [ [x1,y1], [x2, y2], [x3, y3], ... ] that uniquely defines a skyline. A key point is the left endpoint of a horizontal line segment. Note that the last key point, where the rightmost building ends, is merely used to mark the termination of the skyline, and always has zero height. Also, the ground in between any two adjacent buildings should be considered part of the skyline contour.

For instance, the skyline in Figure B should be represented as:[ [2 10], [3 15], [7 12], [12 0], [15 10], [20 8], [24, 0] ].
"""


from typing import List


class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        """
        We could decompose the building in 'start building', 'stop building' events.
        We could sort these events by 'x', and go from left to right:
        - add new height to a data structure when 'start building'
        - remove the matching height when 'stop building'

        This data structure would have to:
        - give us the maximum as fast as possible
        - support adding and remove elements
        => A TreeMap would do the job.
        """

        pass  # TODO


"""
class Solution {
    private abstract class Event {
        public final int position;
        
        public Event(int position) {
            this.position = position;
        }
        
        public abstract void apply(TreeMap<Integer, Integer> skyline);
    }
    
    private class AddEvent extends Event {
        public final int value;
        
        public AddEvent(int position, int value) {
            super(position);
            this.value = value;
        }
        
        public void apply(TreeMap<Integer, Integer> skyline) {
            int count = skyline.getOrDefault(value, 0);
            if (count == -1)
                skyline.remove(value);
            else
                skyline.put(value, count + 1);
        }
        
        public String toString() {
            return "Add{" + position + ";" + value + "}";
        }
    }
    
    private class RemoveEvent extends Event {
        public final int value;
        
        public RemoveEvent(int position, int value) {
            super(position);
            this.value = value;
        }
        
        public void apply(TreeMap<Integer, Integer> skyline) {
            int count = skyline.getOrDefault(value, 0);
            if (count == 1)
                skyline.remove(value);
            else
                skyline.put(value, count - 1);
        }
        
        public String toString() {
            return "Remove{" + position + ";" + value + "}";
        }
    }
    
    public List<List<Integer>> getSkyline(int[][] buildings) {
        /*
        We could decompose the building in 'start building', 'stop building' events.
        We could sort these events by 'x', and go from left to right (already done):
        - add new height to a data structure when 'start building'
        - remove the matching height when 'stop building'
        
        This data structure would have to:
        - give us the maximum as fast as possible
        - support adding and remove elements
        => A TreeMap would do the job
        => An indexed heap (but not really good for remove)
        */
        
        ArrayList<Event> events = new ArrayList<>();
        for (int i = 0; i < buildings.length; i++) {
            int height = buildings[i][2];
            events.add(new AddEvent(buildings[i][0], height));
            events.add(new RemoveEvent(buildings[i][1], height));
        }
        Collections.sort(events, (l, r) -> l.position - r.position);
        // System.out.println(events);
        
        int lastHeight = 0;
        ArrayList<List<Integer>> ret = new ArrayList<>();
        TreeMap<Integer, Integer> skyline = new TreeMap<>();
        for (int i = 0; i < events.size();) {
            Event event = events.get(i);
            // TODO - instead of this ugly shit, group events by position
            while (i < events.size() && events.get(i).position == event.position) {
                events.get(i).apply(skyline);
                i++;
            }
            
            int p = event.position;
            int h = skyline.isEmpty() ? 0 : skyline.lastKey();
            if (h != lastHeight) {
                ret.add(Arrays.asList(event.position, h));
                lastHeight = h;
            }
        }
        return ret;
    }
}
"""

