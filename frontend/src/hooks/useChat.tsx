import { useState, useEffect } from 'react';
import { ChatMessage, Character, Achievement } from '@/types';
import { useToast } from '@/components/ui/use-toast';

interface UseChatProps {
  character?: Character;
}

const useChat = ({ character = "Pam" }: UseChatProps = {}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [currentCharacter, setCurrentCharacter] = useState<Character>(character);
  const [dundiesPoints, setDundiesPoints] = useState<number>(0);
  const [userName, setUserName] = useState<string | null>(null);
  const { toast } = useToast();

  // Load username from localStorage
  useEffect(() => {
    const storedName = localStorage.getItem('officeChatUserName');
    if (storedName) {
      setUserName(storedName);
    }
  }, []);

  // Initialize with a greeting message
  useEffect(() => {
    if (messages.length === 0) {
      const greetingMessage: ChatMessage = {
        id: 'greeting',
        sender: 'character',
        content: userName 
          ? `Hey there, ${userName}! I'm ${currentCharacter}. How's it going?` 
          : `Hi there! I'm ${currentCharacter}. What's your name?`,
        character: currentCharacter,
        timestamp: new Date()
      };
      setMessages([greetingMessage]);
    }
  }, [currentCharacter, userName]);

  // Saves username to localStorage
  const saveUserName = (name: string) => {
    // localStorage.setItem('officeChatUserName', name);
    // setUserName(name);
  };

  // Switch character
  const switchCharacter = (newCharacter: Character) => {
    setCurrentCharacter(newCharacter);
    
    // Play sound effect
    const switchSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-positive-interface-beep-221.mp3');
    switchSound.volume = 0.3;
    switchSound.play();
    
    // Add system message
    const systemMessage: ChatMessage = {
      id: `switch-${Date.now()}`,
      sender: 'character',
      content: `Switching to ${newCharacter}...`,
      character: newCharacter,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, systemMessage]);
    
    // Introduce the new character
    setTimeout(() => {
      const introMessage: ChatMessage = {
        id: `intro-${Date.now()}`,
        sender: 'character',
        content: getIntroMessage(newCharacter, userName || ""),
        character: newCharacter,
        timestamp: new Date()
      };
      setMessages(prev => [...prev.filter(msg => msg.id !== systemMessage.id), introMessage]);
    }, 1000);

    // Show toast message
    toast({
      title: `Now chatting with ${newCharacter}!`,
      description: `${getCharacterQuote(newCharacter)}`,
      className: "font-pixel bg-dundies text-black",
    });
  };

  // Send a message to the API
  const sendMessage = async (content: string) => {
    if (!content.trim()) return;
    
    // Check if it's a command
    if (content.startsWith('/switch ')) {
      const characterName = content.replace('/switch ', '') as Character;
      const validCharacters: Character[] = ["Michael", "Jim", "Pam", "Dwight", "Angela", "Oscar", "Kevin", "Stanley", "Creed", "Toby"];
      
      if (validCharacters.includes(characterName)) {
        switchCharacter(characterName);
        return;
      } else {
        // Invalid character
        toast({
          title: "Invalid character",
          description: "Please choose from the available characters.",
          variant: "destructive",
        });
        return;
      }
    }

    // Check if the user is providing their name for the first time
    if (!userName && messages.length === 1 && messages[0].content.includes("What's your name?")) {
      saveUserName(content);
      
      const welcomeMessage: ChatMessage = {
        id: `welcome-${Date.now()}`,
        sender: 'character',
        content: `Nice to meet you, ${content}! I'm ${currentCharacter} from Dunder Mifflin. How can I help you today?`,
        character: currentCharacter,
        timestamp: new Date()
      };
      
      // Add user message
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        sender: 'user',
        content: content,
        character: currentCharacter,
        timestamp: new Date()
      };
      
      setMessages([...messages, userMessage, welcomeMessage]);
      setDundiesPoints(prev => prev + 1);
      return;
    }
    
    // Regular message flow
    // Add user message
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      content: content,
      character: currentCharacter,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    
    // Play typing sound
    const typingSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-typewriter-soft-click-1125.mp3');
    typingSound.volume = 0.2;
    typingSound.play();
    
    setLoading(true);
    
    try {
      console.log(userName, currentCharacter, content, messages);
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_name: userName || "Guest",
          character: currentCharacter,
          query: content,
          memory: messages.map(m => ({
            type: m.sender === "user" ? "human" : "ai",
            content: m.content
          }))
    
        }),
      });
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();

            
      // For now, simulate a response:
      // await new Promise(resolve => setTimeout(resolve, 1500));
      
      // const mockResponse = getMockResponse(content, currentCharacter);
      const mockResponse = data.response;

      
      const characterMessage: ChatMessage = {
        id: `char-${Date.now()}`,
        sender: 'character',
        content: mockResponse,
        character: currentCharacter,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, characterMessage]);
      
      // Increase Dundies Points
      setDundiesPoints(prev => {
        const newPoints = prev + 1;
        
        // Check for achievements
        if (newPoints === 5) {
          triggerAchievement({
            id: 'first-five',
            title: 'Casual Friday',
            description: 'Chatted 5 times! You\'re getting the hang of it!',
            icon: '🏆'
          });
        } else if (newPoints === 10) {
          triggerAchievement({
            id: 'assistant-regional-manager',
            title: 'Assistant TO THE Regional Manager',
            description: 'Reached 10 Dundies Points! Michael almost knows your name now.',
            icon: '📋'
          });
        } else if (newPoints === 20) {
          triggerAchievement({
            id: 'pretzel-day',
            title: 'Pretzel Day Pro',
            description: 'Reached 20 Dundies Points! Stanley would be proud.',
            icon: '🥨'
          });
        }
        
        // Check for Easter eggs - every 5 messages
        if (newPoints % 5 === 0) {
          triggerEasterEgg();
        }
        
        return newPoints;
      });
      
    } catch (error) {
      console.error('Error sending message:', error);
      toast({
        title: "Error",
        description: "Failed to get a response. The server might be down.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  // Trigger achievement
  const triggerAchievement = (achievement: Achievement) => {
    // Play achievement sound
    const achievementSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-arcade-game-complete-or-approved-mission-205.mp3');
    achievementSound.volume = 0.4;
    achievementSound.play();
    
    // Show achievement toast
    toast({
      title: `Achievement Unlocked: ${achievement.title}`,
      description: achievement.description,
      className: "font-pixel bg-dundies text-black border-4 border-black",
    });
  };

  // Trigger easter egg
  const triggerEasterEgg = () => {
    const easterEggs = [
      "That's what she said!",
      "Bears. Beets. Battlestar Galactica.",
      "I'm not superstitious, but I am a little stitious.",
      "Identity theft is not a joke, Jim!",
      "Why waste time say lot word when few word do trick?"
    ];
    
    const randomEgg = easterEggs[Math.floor(Math.random() * easterEggs.length)];
    
    // Play easter egg sound
    const eggSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-unlock-game-notification-253.mp3');
    eggSound.volume = 0.4;
    eggSound.play();
    
    // Show easter egg toast
    toast({
      title: "Office Easter Egg!",
      description: randomEgg,
      className: "font-pixel bg-schrute text-white border-4 border-black",
    });
  };

  return {
    messages,
    loading,
    currentCharacter,
    dundiesPoints,
    userName,
    sendMessage,
    switchCharacter
  };
};

// Helper functions
const getIntroMessage = (character: Character, userName: string): string => {
  const greetings: Record<Character, string> = {
    Michael: userName ? `Hey ${userName}! That's what she said! I'm Michael Scott, World's Best Boss. Let's talk about some awesome stuff!` : "I'm Michael Scott, World's Best Boss! What's your name?",
    Jim: userName ? `Hey there, ${userName}. Jim Halpert here. How's it going? Seen Dwight's stapler lately?` : "Jim Halpert here. What's your name?",
    Pam: userName ? `Hi ${userName}! I'm Pam. Need me to transfer your call or... just chat?` : "This is Pam Beesly at reception. Who am I speaking with?",
    Dwight: userName ? `${userName}! Listen carefully. I'm Dwight Schrute, Assistant Regional Manager. What's your business here?` : "Identify yourself immediately. Dwight Schrute, Assistant Regional Manager.",
    Angela: userName ? `Hello, ${userName}. Angela Martin, head of accounting. Let's keep this conversation professional.` : "This is Angela Martin. And you are?",
    Oscar: userName ? `Hi ${userName}, Oscar Martinez here. How can I help you today?` : "Oscar Martinez speaking. May I ask who's calling?",
    Kevin: userName ? `Hey ${userName}... I'm Kevin... I like cookies... and M&Ms... and you?` : "Kevin Malone... Who are you?",
    Stanley: userName ? `${userName}? Stanley Hudson. Make it quick, my crossword puzzle isn't going to solve itself.` : "Stanley Hudson. And you are?",
    Creed: userName ? `Hey... ${userName}? Sure, that works. I'm Creed. Have we met before? You look familiar.` : "Name's Creed Bratton. What's your name again? Wait, don't tell me...",
    Toby: userName ? `Hi there, ${userName}. Toby from HR. Nice to meet you. How can I help today?` : "This is Toby Flenderson from HR. What's your name?"
  };
  
  return greetings[character];
};

const getCharacterQuote = (character: Character): string => {
  const quotes: Record<Character, string> = {
    Michael: "I'm not superstitious, but I am a little stitious.",
    Jim: "Bears. Beets. Battlestar Galactica.",
    Pam: "There's a lot of beauty in ordinary things. Isn't that kind of the point?",
    Dwight: "Identity theft is not a joke, Jim!",
    Angela: "I don't have a headache. I'm just preparing.",
    Oscar: "Actually...",
    Kevin: "Why waste time say lot word when few word do trick?",
    Stanley: "Did I stutter?",
    Creed: "I've been involved in a number of cults, both as a leader and a follower.",
    Toby: "Why you gotta be so mean to me?"
  };
  
  return quotes[character];
};

const getMockResponse = (query: string, character: Character): string => {
  // Simple mock responses - in real app, these would come from the API
  const lowerQuery = query.toLowerCase();
  
  if (lowerQuery.includes("hello") || lowerQuery.includes("hi ")) {
    const greetings: Record<Character, string> = {
      Michael: "Hello! I'm Michael Scott, Regional Manager and World's Best Boss (I have a mug that proves it).",
      Jim: "Hey there! Jim Halpert. I'd shake your hand but... you know, digital barriers.",
      Pam: "Hi! Pam Beesly speaking. How can I help you today?",
      Dwight: "Greetings. Dwight Schrute, Assistant Regional Manager. State your purpose.",
      Angela: "Hello. Angela Martin speaking.",
      Oscar: "Hello there. Oscar Martinez from accounting.",
      Kevin: "Heyyyyy... I'm Kevin... from the office...",
      Stanley: "Hello. Stanley Hudson speaking. Make it quick.",
      Creed: "Hey, chief. The name's Creed Bratton. I may have stolen that identity, though.",
      Toby: "Hi there. Toby Flenderson, HR. Nice to meet you."
    };
    return greetings[character];
  }
  
  if (lowerQuery.includes("how are you")) {
    const responses: Record<Character, string> = {
      Michael: "Amazing! Just finished watching Threat Level Midnight for the 50th time. Pure cinema!",
      Jim: "Not bad. Planning my next prank on Dwight. Any suggestions?",
      Pam: "I'm good! Just answering phones and doodling when no one's looking.",
      Dwight: "Optimal. My beet farm yielded 18% more crop this quarter. I've also reorganized my survival shelter.",
      Angela: "I'm fine. My cats are well, which is all that matters.",
      Oscar: "I'm well, thank you for asking. Just trying to keep the accounting department functioning.",
      Kevin: "Good... hungry... thinking about cookies... and chili...",
      Stanley: "It's not Pretzel Day, so just average.",
      Creed: "Who wants to know? Are they looking for me? Tell them I've been in Mexico.",
      Toby: "I'm okay. Michael didn't yell at me today, so that's a win."
    };
    return responses[character];
  }
  
  if (lowerQuery.includes("joke") || lowerQuery.includes("funny")) {
    const jokes: Record<Character, string> = {
      Michael: "Why did the scarecrow win an award? Because he was outstanding in his field! That's what she said!",
      Jim: "What's the most important thing to learn in chemistry? Never lick the spoon.",
      Pam: "What did the janitor say when he jumped out of the closet? 'Supplies!'",
      Dwight: "Joke? I don't joke. But here's a fact: bears can climb faster than they can run.",
      Angela: "I don't tell jokes. Frivolous humor leads to moral decay.",
      Oscar: "What do accountants use to solve their problems? Spread sheets.",
      Kevin: "Why did the cookie go to the doctor? It was feeling crumbly... hehe...",
      Stanley: "Why did I come to work today? That's the joke.",
      Creed: "I saw a guy walking a dog the other day. I said to him, 'Does your dog bite?' He said, 'No.' So I pet it and it bit me! I said, 'I thought you said your dog doesn't bite!' He said, 'That's not my dog.'",
      Toby: "What's the difference between me and Michael Scott? HR would actually listen if you complained about me..."
    };
    return jokes[character];
  }
  
  // Default responses if no patterns match
  const defaults: Record<Character, string> = {
    Michael: "That's what she said! Wait, does that work here? Anyway, let's circle back and touch base on this fascinating conversation.",
    Jim: "Interesting. *Looks at camera with raised eyebrows*",
    Pam: "Sorry, I was just drawing something. What were you saying?",
    Dwight: "Your inquiry lacks sufficient detail for me to formulate an optimal response. Please rephrase with more tactical precision.",
    Angela: "*Judging silently* I suppose we could discuss this further, if we must.",
    Oscar: "Actually, that's an interesting point. Let me explain my perspective...",
    Kevin: "Cool... cool cool cool... Hey, do you have any snacks?",
    Stanley: "*Sighs deeply* Are we done here? My crossword puzzle isn't going to finish itself.",
    Creed: "That reminds me of the time I was in an iron lung with Muhammad Ali. Strange days.",
    Toby: "That's a good point. I'd like to hear more, unlike when I try to talk to Michael."
  };
  
  return defaults[character];
};

export default useChat;
