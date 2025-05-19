
import { ChatMessage } from '@/types';

interface ChatBubbleProps {
  message: ChatMessage;
}

const ChatBubble = ({ message }: ChatBubbleProps) => {
  const isUser = message.sender === 'user';
  
  // Format timestamp
  const formatTime = (date: Date) => {
    return new Date(date).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} my-2`}>
      <div 
        className={`
          max-w-[80%] sm:max-w-[70%] 
          ${isUser 
              ? 'bg-pretzel bg-opacity-70 text-black mr-2' 
              : 'bg-schrute bg-opacity-70 text-white ml-2'
           }
          p-3 pixel-border rounded
          font-pixel
        `}
      >
        {!isUser && (
          <div className="font-bold mb-1 text-dundies">
            {message.character}
          </div>
        )}
        <p className="whitespace-pre-wrap">{message.content}</p>
        <div className={`text-xs mt-1 text-opacity-80 ${isUser ? 'text-black' : 'text-white'}`}>
          {formatTime(message.timestamp)}
        </div>
      </div>
    </div>
  );
};

export default ChatBubble;
