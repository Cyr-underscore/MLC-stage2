import React from 'react';
import '../style/modal.css';

interface ModalProps {
    isOpen: boolean;
    onClose?: () => void;
    children: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({ isOpen, onClose, children }) => {
    if (!isOpen) return null;

    return (
        <div className="modal-overlay">
            <div className="modal">
                {onClose && (
                    <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">
                        &times;
                    </button>
                )}
                {children}
            </div>
        </div>
    );
};

export default Modal; 
