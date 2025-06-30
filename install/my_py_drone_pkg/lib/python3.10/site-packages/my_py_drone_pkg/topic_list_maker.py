import rclpy
from rclpy.node import Node

def list_topics():
    """
    List all available topics in the ROS 2 system
    """
    # Initialize ROS 2 client library
    rclpy.init()
    
    # Create a node for topic discovery
    node = Node('topic_lister')
    
    try:
        # Get topic names and types
        topic_names_and_types = node.get_topic_names_and_types()
        
        # Print topics
        print("Available Topics:")
        print("-" * 40)
        for topic, types in topic_names_and_types:
            print(f"Topic: {topic}")
            print(f"  Types: {', '.join(types)}")
            print("-" * 40)
    
    except Exception as e:
        print(f"Error listing topics: {e}")
    
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()

def main():
    list_topics()

if __name__ == '__main__':
    main()